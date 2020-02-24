import itertools
import os
import xml.etree.ElementTree as ET
from itertools import compress

import numpy

from csep.utils.calc import bin1d_vec
from csep.utils.basic_types import Polygon
from csep.utils.constants import CSEP_MW_BINS
from csep.utils.scaling_relationships import WellsAndCoppersmith

class CartesianGrid2D:
    """Represents a 2D cartesian gridded region.

    The class provides functions to query onto an index 2D Cartesian grid and maintains a mapping between space coordinates defined
    by polygons and the index into the polygon array.

    Custom regions can be easily created by using the from_polygon classmethod. This function will accept an arbitrary closed
    polygon and return a CartesianGrid class with only points inside the polygon to be valid.
    """
    def __init__(self, polygons, dh, name='Generic CartesianGrid2D'):
        self.polygons = polygons
        self.dh = dh
        self.name = name
        a, xs, ys = build_bitmask_vec(self.polygons, dh)
        # bitmask is 2d numpy array with 2d shape (xs, ys), dim=0 is the mapping from 2d to polygon dim=1 maps the index
        # note: might consider changing this, but it requires less constraints on how the polygons are defined.
        self.bitmask = a
        # index values of polygons array into the 2d cartesian grid, based on the midpoint.
        self.xs = xs
        self.ys = ys

    @property
    def num_nodes(self):
        return len(self.polygons)

    def get_index_of(self, lons, lats):
        """
        Returns the index of lons, lats in self.polygons
        Args:
            lons: ndarray-like
            lats: ndarray-like

        Returns:
            idx: ndarray-like
        """
        idx = bin1d_vec(lons, self.xs)
        idy = bin1d_vec(lats, self.ys)
        return self.bitmask[idy, idx, 1].astype(numpy.int)

    def get_location_of(self, idx):
        """
        Returns the polygon associated with the index idx.

        Args:
            idx: index of polygon in region

        Returns:
            Polygon

        """
        idx = list(idx)
        midpoints = numpy.array([self.polygons[int(ix)].centroid() for ix in idx])
        return midpoints[:,0], midpoints[:,1]

    def get_masked(self, lons, lats):
        """Returns bool array lons and lats are not included in the spatial region.

        .. note:: The ordering of lons and lats should correspond to the ordering of the lons and lats in the catalog.

        Args:
            lons: array-like
            lats: array-like

        Returns:
            idx: array-like
        """

        idx = bin1d_vec(lons, self.xs)
        idy = bin1d_vec(lats, self.ys)
        return self.bitmask[idy, idx, 0].astype(bool)

    def get_cartesian(self, data):
        """Returns 2d ndrray representation of the data set, corresponding to the bounding box.

        Args:
            data:
        """
        assert len(data) == len(self.polygons)
        results = numpy.zeros(self.bitmask.shape[:2])
        ny = len(self.ys)
        nx = len(self.xs)
        for i in range(ny):
            for j in range(nx):
                if self.bitmask[i, j, 0] == 0:
                    idx = int(self.bitmask[i, j, 1])
                    results[i, j] = data[idx]
                else:
                    results[i, j] = numpy.nan
        return results

    def get_bbox(self):
        return (self.xs.min(), self.xs.max(), self.ys.min(), self.ys.max())

    def midpoints(self):
        return numpy.array([poly.centroid() for poly in self.polygons])

    def origins(self):
        return numpy.array([poly.origin for poly in self.polygons])

    def to_dict(self):
        adict = {
            'name': self.name,
            'dh': self.dh,
            'polygons': [{'lat': origin[1], 'lon': origin[0]} for origin in self.polygons.origin]
        }
        return adict

    @classmethod
    def from_dict(cls, adict):
        raise NotImplementedError("Todo!")


def grid_spacing(vertices):
    """
    Figures out the length and

    Args:
        vertices: Vertices describe a single node in grid.

    Returns:
        dh: grid spacing

    """
    # get first two vertices
    a = vertices[0]
    b = vertices[1]
    # compute both differences, because unless point is the same one is bound to be the dh
    d1 = numpy.abs(b[0] - a[0])
    d2 = numpy.abs(b[1] - a[1])
    dh = numpy.max([d1, d2])
    # this would happen if the same point is repeated twice
    if dh == 0:
        raise ValueError("Problem computing grid spacing cannot be zero.")
    return dh

def california_relm_region(filepath=None, dh=0.1):
    """
    Takes a CSEP1 XML file and returns a 'region' which is a list of polygons. This region can
    be used to create gridded datasets for earthquake forecasts.

    Args:
        filepath: filepath to CSEP1 style XML forecast template
        dh: grid spacing in lat / lon. (Assumed to be equal).

    Returns:
        list of polygons (region)

    """
    # todo: allow user to specifiy grid spacing
    if filepath is None:
        # use default file path from python pacakge
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'csep-forecast-template-M5.xml')
    csep_template = os.path.expanduser(filepath)
    origins = parse_csep_template(csep_template)
    origins = increase_grid_resolution(origins, dh, 4)
    dh = dh / 4
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name='California RELM CartesianGrid2D')
    return relm_region


def global_region(dh=0.1, name="global"):
    """ Creates a global region used for evaluating gridded forecasts on the global scale.

    The gridded region corresponds to the

    Args:
        dh:

    Returns:
        csep.utils.CartesianGrid2D:
    """
    # generate latitudes
    lats = numpy.arange(-90.0, 89.9 + dh/2, dh)
    lons = numpy.arange(-180, 179.9 + dh/2, dh)
    coords = itertools.product(lons,lats)
    bboxes = compute_vertices(coords, dh)
    region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)
    return region


def parse_csep_template(xml_filename):
    """
    Reads CSEP XML template file and returns the lat/lon values
    for the forecast.

    Returns:
        list of tuples where tuple is (lon, lat)
    """
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    points = []
    for cell in root.iter('{http://www.scec.org/xml-ns/csep/forecast/0.1}cell'):
        points.append((float(cell.attrib['lon']), float(cell.attrib['lat'])))
    return points

def increase_grid_resolution(points, dh, factor):
    """
    takes a set of origin points and returns a new set with higher grid resolution. assumes the origin point is in the
    lower left corner. the new dh is dh / factor.

    Args:
        points: list of (lon,lat) tuples
        dh: old grid spacing
        factor: amount to reduce

    Returns:
        points: list of (lon,lat) tuples with spacing dh / scale

    """
    # short-circuit recursion
    if factor == 1:
        return points

    # handle edge cases
    assert factor % 2 == 0
    assert factor >= 1

    # first start out
    new_points = set()
    new_dh = dh / 2
    for point in points:
        bbox = compute_vertex(point, new_dh)
        for pnt in bbox:
            new_points.add(pnt)
    # call function again with new_points, new_dh, new_factor
    new_factor = factor / 2
    return increase_grid_resolution(list(new_points), new_dh, new_factor)

def compute_vertex(origin_point, dh, tol=numpy.finfo(float).eps):
    """
    Computes the bounding box of a rectangular polygon given its origin points and spacing dh.

    Args:
        origin_points: list of tuples, where tuple is (x, y)
        dh: spacing
        tol: used to eliminate overlapping polygons in the case of a rectangular mesh, defaults to
             the machine tolerance.

    Returns:
        list of polygon edges

    """
    bbox = ((origin_point[0], origin_point[1]),
            (origin_point[0], origin_point[1] + dh - tol),
            (origin_point[0] + dh - tol, origin_point[1] + dh - tol),
            (origin_point[0] + dh - tol, origin_point[1]))
    return bbox

def compute_vertices(origin_points, dh, tol=numpy.finfo(float).eps):
    """
    Wrapper function to compute vertices for multiple points. Default tolerance is set to machine precision
    of floating point number.

    Args:
        origin_points: 2d ndarray

    Notes:
        (x,y) should be accessible like:
        >>> x_coords = origin_points[:,0]
        >>> y_coords = origin_points[:,1]

    """
    return list(map(lambda x: compute_vertex(x, dh, tol=tol), origin_points))

def build_bitmask_vec(polygons, dh):
    """
    same as build bitmask but using vectorized calls to bin1d
    """
    # build bounding box of set of polygons based on origins
    nd_origins = numpy.array([poly.origin for poly in polygons])
    bbox = [(numpy.min(nd_origins[:, 0]), numpy.min(nd_origins[:, 1])),
            (numpy.max(nd_origins[:, 0]), numpy.max(nd_origins[:, 1]))]

    # get midpoints for hashing
    midpoints = numpy.array([poly.centroid() for poly in polygons])

    # compute nx and ny
    nx = numpy.rint((bbox[1][0] - bbox[0][0]) / dh)
    ny = numpy.rint((bbox[1][1] - bbox[0][1]) / dh)

    # set up grid of bounding box
    xs = dh * numpy.arange(nx + 1) + bbox[0][0]
    ys = dh * numpy.arange(ny + 1) + bbox[0][1]

    # set up mask array, 1 is index 0 is mask
    a = numpy.ones([len(ys), len(xs), 2])

    # bin1d returns the index of polygon within the cartesian grid
    idx = bin1d_vec(midpoints[:, 0], xs)
    idy = bin1d_vec(midpoints[:, 1], ys)

    for i in range(len(polygons)):

        # store index of polygon
        a[idy[i], idx[i], 1] = int(i)

        # build bitmask, if statement is confusing.
        if idx[i] >= 0 and idy[i] >= 0:
            a[idy[i], idx[i], 0] = 0

    return a, xs, ys

def bin_catalog_spatio_magnitude_counts(lons, lats, mags, n_poly, bitmask, binx, biny):
    """
    Returns a list of event counts as ndarray with shape (n_poly, n_cat) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the bitmask, we store the mapping between the index of n_poly and
    that polygon in the bitmask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    Eventually, we can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """

    # index in cartesian grid for events in catalog. note, this has a different index than the
    # vector of polygons. this mapping is stored in [:,:,1] index of bitmask
    ny, nx, _ = bitmask.shape
    # index in 2d grid
    idx = bin1d_vec(lons, binx)
    idy = bin1d_vec(lats, biny)
    mags = bin1d_vec(mags, CSEP_MW_BINS)
    # start with zero event counts in each bin
    event_counts = numpy.zeros((n_poly, len(CSEP_MW_BINS)))
    # does not seem that we can vectorize this part
    for i in range(idx.shape[0]):
        # we store the index of that polygon in array [:, :, 1], flag is [:,:,0]
        if not bitmask[idy[i], idx[i], 0]:
            # getting spatial bin from bitmask
            hash_idx = int(bitmask[idy[i], idx[i], 1])
            mag_idx = mags[i]
            # update event counts in that polygon
            event_counts[(hash_idx, mag_idx)]+=1
    return event_counts

def bin_catalog_spatial_counts(lons, lats, n_poly, bitmask, binx, biny):
    """
    Returns a list of event counts as ndarray with shape (n_poly) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the bitmask, we store the mapping between the index of n_poly and
    that polygon in the bitmask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    We can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """
    ai, bi = binx, biny
    # index in cartesian grid for events in catalog. note, this has a different index than the
    # vector of polygons. this mapping is stored in [:,:,1] index of bitmask
    idx = bin1d_vec(lons, ai)
    idy = bin1d_vec(lats, bi)
    event_counts = numpy.zeros(n_poly)
    # [:,:,1] is a mapping from the polygon array to cartesian grid
    hash_idx=bitmask[idy,idx,1].astype(int)
    # this line seems redundant. need unit testing before merging to master.
    hash_idx[bitmask[idy,idx,0] != 0] = 0
    numpy.add.at(event_counts, hash_idx, 1)
    return event_counts

def bin_catalog_probability(lons, lats, n_poly, bitmask, binx, biny):
    """
    Returns a list of event counts as ndarray with shape (n_poly) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the bitmask, we store the mapping between the index of n_poly and
    that polygon in the bitmask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    We can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """
    ai, bi = binx, biny
    # index in cartesian grid for events in catalog. note, this has a different index than the
    # vector of polygons. this mapping is stored in [:,:,1] index of bitmask
    idx = bin1d_vec(lons, ai)
    idy = bin1d_vec(lats, bi)
    event_counts = numpy.zeros(n_poly)
    # [:,:,1] is a mapping from the polygon array to cartesian grid
    hash_idx=bitmask[idy,idx,1].astype(int)
    # this line seems redundant, because lons/lats should be removed if outside of polygon. need unit testing before merging to master.
    hash_idx[bitmask[idy,idx,0] != 0] = 0
    # dont accumulate just set to one
    event_counts[hash_idx] = 1
    return event_counts

def masked_region(region, polygon):
    """
    build a new region based off the coordinates in the polygon. warning: light weight and no error checking.
    Args:
        region: CartesianGrid2D object
        polygon: Polygon object

    Returns:
        new_region: CartesianGrid2D object
    """
    # contains is true if spatial cell in region is inside the polygon
    contains = polygon.contains(region.midpoints())
    # compress only returns elements that are true, effectively removing elements outside of the polygons
    new_polygons = list(compress(region.polygons, contains))
    # create new region with the spatial cells inside the polygon
    return CartesianGrid2D(new_polygons, region.dh)

def generate_aftershock_region(mainshock_mw, mainshock_lon, mainshock_lat, num_radii=3):
    # filter to aftershock radius
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(mainshock_mw) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((mainshock_lon, mainshock_lat),
                                                          num_radii * rupture_length, num_points=100)
    aftershock_region = masked_region(california_relm_region(), aftershock_polygon)
    return aftershock_region
