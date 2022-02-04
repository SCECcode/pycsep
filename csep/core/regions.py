# Python imports
import itertools
import os
from itertools import compress
from xml.etree import ElementTree as ET

# Third-party imports
import numpy
import numpy as np
import mercantile
import multiprocessing as mp
import shapely
from functools import partial


# PyCSEP imports
from csep.utils.calc import bin1d_vec, cleaner_range, first_nonnan, last_nonnan
from csep.utils.scaling_relationships import WellsAndCoppersmith

from csep.models import Polygon


def california_relm_collection_region(dh_scale=1, magnitudes=None, name="relm-california-collection", use_midpoint=True):
    """ Return collection region for California RELM testing region

    Args:
        dh_scale (int): factor of two multiple to change the grid size
        mangitudes (array-like): array representing the lower bin edges of the magnitude bins
        name (str): human readable identifer
        use_midpoints (bool): if true, treat values in file as midpoints. default = true.

    Returns:
        :class:`csep.core.spatial.CartesianGrid2D`

    Raises:
        ValueError: dh_scale must be a factor of two

    """
    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

    # we can hard-code the dh because we hard-code the filename
    dh = 0.1
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'RELMCollectionArea.dat')
    points = numpy.loadtxt(filepath)
    if use_midpoint:
        origins = numpy.array(points) - dh / 2
    else:
        origins = numpy.array(points)

    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        relm_region.magnitudes = magnitudes

    return relm_region

def california_relm_region(dh_scale=1, magnitudes=None, name="relm-california", use_midpoint=True):
    """
    Returns class representing California testing region.

    This region can
    be used to create gridded datasets for earthquake forecasts. The XML file appears to use the
    midpoint, and the .dat file uses the origin in the "lower left" corner.

    Args:
        dh_scale: can resample this grid by factors of 2

    Returns:
        :class:`csep.core.spatial.CartesianGrid2D`

    Raises:
        ValueError: dh_scale must be a factor of two

    """

    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

        # use default file path from python package
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'csep-forecast-template-M5.xml')
    csep_template = os.path.expanduser(filepath)
    points, dh = parse_csep_template(csep_template)
    if use_midpoint:
        origins = numpy.array(points) - dh / 2
    else:
        origins = numpy.array(points)

    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        relm_region.magnitudes = magnitudes

    return relm_region

def italy_csep_region(dh_scale=1, magnitudes=None, name="csep-italy", use_midpoint=True):
    """
        Returns class representing Italian testing region.

        This region can be used to create gridded datasets for earthquake forecasts. The region is defined by the
        file 'forecast.italy.M5.xml' and contains a spatially gridded region with 0.1° x 0.1° cells.

        Args:
            dh_scale: can resample this grid by factors of 2
            magnitudes (array-like): bin edges for magnitudes. if provided, will be bound to the output region class.
                                     this argument provides a short-cut for creating space-magnitude regions.
            name (str): human readable identify given to the region
            use_midpoint (bool): if true, treat values in file as midpoints. default = true.

        Returns:
            :class:`csep.core.spatial.CartesianGrid2D`

        Raises:
            ValueError: dh_scale must be a factor of two

    """
    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

        # use default file path from python package
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'forecast.italy.M5.xml')
    csep_template = os.path.expanduser(filepath)
    points, dh = parse_csep_template(csep_template)
    if use_midpoint:
        origins = numpy.array(points) - dh / 2
    else:
        origins = numpy.array(points)


    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    italy_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        italy_region.magnitudes = magnitudes

    return italy_region

def italy_csep_collection_region(dh_scale=1, magnitudes=None, name="csep-italy-collection", use_midpoint=True):
    """ Return collection region for Italy CSEP collection region

        Args:
            dh_scale (int): factor of two multiple to change the grid size
            mangitudes (array-like): array representing the lower bin edges of the magnitude bins
            name (str): human readable identifer
            use_midpoint (bool): if true, treat values in file as midpoints. default = true.

        Returns:
            :class:`csep.core.spatial.CartesianGrid2D`

        Raises:
            ValueError: dh_scale must be a factor of two
    """
    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

    # we can hard-code the dh because we hard-code the filename
    dh = 0.1
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'italy.collection.nodes.dat')
    points = numpy.loadtxt(filepath)
    if use_midpoint:
        origins = numpy.array(points) - dh / 2
    else:
        origins = numpy.array(points)


    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        relm_region.magnitudes = magnitudes

    return relm_region

def global_region(dh=0.1, name="global", magnitudes=None):
    """ Creates a global region used for evaluating gridded forecasts on the global scale.

    The gridded region corresponds to the

    Args:
        dh:

    Returns:
        csep.utils.CartesianGrid2D:
    """
    # generate latitudes

    lons = cleaner_range(-180.0, 179.9, dh)
    lats = cleaner_range(-90, 89.9, dh)
    coords = itertools.product(lons,lats)
    region = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(coords, dh)], dh, name=name)
    if magnitudes is not None:
        region.magnitudes = magnitudes
    return region

def magnitude_bins(start_magnitude, end_magnitude, dmw):
    """ Returns array holding magnitude bin edges.

    The output from this function is monotonically increasing and equally spaced bin edges that can represent magnitude
    bins.

     Args:
        start_magnitude (float)
        end_magnitude (float)
        dmw (float): magnitude spacing

    Returns:
        bin_edges (numpy.ndarray)
    """
    # convert to integers to prevent accumulating floating point errors
    const = 10000
    start = numpy.floor(const * start_magnitude)
    end = numpy.floor(const * end_magnitude)
    d = const * dmw
    return numpy.arange(start, end + d / 2, d) / const

def create_space_magnitude_region(region, magnitudes):
    """Simple wrapper to create space-magnitude region """
    if not (isinstance(region, CartesianGrid2D) or isinstance(region, QuadtreeGrid2D))  :
        raise TypeError("region must be CartesianGrid2D")
    # bind to region class
    if magnitudes is None:
        raise ValueError("magnitudes should not be None if creating space-magnitude region.")
    region.magnitudes = magnitudes
    region.num_mag_bins = len(region.magnitudes)
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

    # get cell spacing
    data = root.find('{http://www.scec.org/xml-ns/csep/forecast/0.1}forecastData')
    dh_elem = data.find('{http://www.scec.org/xml-ns/csep/forecast/0.1}defaultCellDimension')
    dh_lat = float(dh_elem.attrib['latRange'])
    dh_lon = float(dh_elem.attrib['lonRange'])

    if not numpy.isclose(dh_lat, dh_lon):
        raise ValueError("dh_lat must equal dh_lon. grid needs to be regular.")

    return points, dh_lat

def increase_grid_resolution(points, dh, factor):
    """
    Takes a set of origin points and returns a new set with higher grid resolution. assumes the origin point is in the
    lower left corner. the new dh is dh / factor. This implementation requires that the decimation factor be a multiple of 2.

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

def masked_region(region, polygon):
    """
    Build a new region based off the coordinates in the polygon.

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

def generate_aftershock_region(mainshock_mw, mainshock_lon, mainshock_lat, num_radii=3, region=california_relm_region, **kwargs):
    """ Creates a spatial region around a given epicenter

    The method uses the Wells and Coppersmith scaling relationship to determine the average fault length and creates a
    circular region centered at (mainshock_lon, mainshock_lat) with radius = num_radii.

    Args:
        mainshock_mw (float): magnitude of mainshock
        mainshock_lon (float): epicentral longitude
        mainshock_lat (float): epicentral latitude
        num_radii (float/int): number of radii of circular region
        region (callable): returns :class:`csep.utils.spatial.CartesianGrid2D`
        **kwargs (dict): passed to region callable

    Returns:
        :class:`csep.utils.spatial.CartesianGrid2D`

    """
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(mainshock_mw) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((mainshock_lon, mainshock_lat),
                                                          num_radii * rupture_length, num_points=100)
    aftershock_region = masked_region(region(**kwargs), aftershock_polygon)
    return aftershock_region

def grid_spacing(vertices):
    """
    Figures out the length and

    Args:
        vertices: Vertices describe a single node in grid.

    Returns:
        dh: grid spacing

    Raises:
        ValueError

    """
    # get first two vertices
    a = vertices[0]
    b = vertices[1]
    # compute both differences, because unless point is the same one is bound to be the dh
    d1 = numpy.abs(b[0] - a[0])
    d2 = numpy.abs(b[1] - a[1])
    if not numpy.allclose(d1, d2):
        raise ValueError("grid spacing must be regular for cartesian grid.")
    dh = numpy.max([d1, d2])
    # this would happen if the same point is repeated twice
    if dh == 0:
        raise ValueError("Problem computing grid spacing cannot be zero.")
    return dh

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
        #>>> x_coords = origin_points[:,0]
        #>>> y_coords = origin_points[:,1]

    """
    return list(map(lambda x: compute_vertex(x, dh, tol=tol), origin_points))

def _bin_catalog_spatio_magnitude_counts(lons, lats, mags, n_poly, mask, idx_map, binx, biny, mag_bins, tol=0.00001):
    """
    Returns a list of event counts as ndarray with shape (n_poly, n_cat) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the mask, we store the mapping between the index of n_poly and
    that polygon in the mask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    Eventually, we can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """

    # index in cartesian grid for events in data. note, this has a different index than the
    # vector of polygons. this mapping is stored in [:,:,1] index of mask
    # index in 2d grid
    idx = bin1d_vec(lons, binx)
    idy = bin1d_vec(lats, biny)
    mag_idxs = bin1d_vec(mags, mag_bins, tol=tol, right_continuous=True)
    # start with zero event counts in each bin
    event_counts = numpy.zeros((n_poly, len(mag_bins)))
    # does not seem that we can vectorize this part
    skipped = []
    for i in range(idx.shape[0]):
        if not mask[idy[i], idx[i]] and idy[i] != -1 and idx[i] != -1 and mag_idxs[i] != -1:
            # getting spatial bin from mask
            hash_idx = int(idx_map[idy[i], idx[i]])
            mag_idx = mag_idxs[i]
            # update event counts in that polygon
            event_counts[(hash_idx, mag_idx)] += 1
        else:
            skipped.append((lons[i], lats[i], mags[i]))

    return event_counts, skipped

def _bin_catalog_spatial_counts(lons, lats, n_poly, mask, idx_map, binx, biny):
    """
    Returns a list of event counts as ndarray with shape (n_poly) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the mask, we store the mapping between the index of n_poly and
    that polygon in the mask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    We can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """
    ai, bi = binx, biny
    # will return negative
    idx = bin1d_vec(lons, ai)
    idy = bin1d_vec(lats, bi)
    # bin1d returns -1 if outside the region
    # todo: think about how to change this behavior for less confusions, bc -1 is an actual value that can be chosen
    bad = (idx == -1) | (idy == -1) | (mask[idy,idx] == 1)
    # this can be memory optimized by keeping short list and storing index, only for case where n/2 events
    event_counts = numpy.zeros(n_poly)
    # selecting the indexes into polygons correspoding to lons and lats within the grid
    hash_idx = idx_map[idy[~bad],idx[~bad]].astype(int)
    # aggregate in counts
    numpy.add.at(event_counts, hash_idx, 1)
    return event_counts

def _bin_catalog_probability(lons, lats, n_poly, mask, idx_map, binx, biny):
    """
    Returns a list of event counts as ndarray with shape (n_poly) where each value
    represents the event counts within the polygon.

    Using [:, :, 1] index of the mask, we store the mapping between the index of n_poly and
    that polygon in the mask. Additionally, the polygons are ordered such that the index of n_poly
    in the result corresponds to the index of the polygons.

    We can make a structure that could contain both of these, but the trade-offs will need
    to be compared against performance.
    """
    ai, bi = binx, biny
    # returns -1 if outside of the bbox
    idx = bin1d_vec(lons, ai)
    idy = bin1d_vec(lats, bi)
    bad = (idx == -1) | (idy == -1) | (mask[idy, idx] == 1)
    event_counts = numpy.zeros(n_poly)
    # [:,:,1] is a mapping from the polygon array to cartesian grid
    hash_idx = idx_map[idy[~bad],idx[~bad]].astype(int)
    # dont accumulate just set to one for probability
    event_counts[hash_idx] = 1
    return event_counts

class CartesianGrid2D:
    """Represents a 2D cartesian gridded region.

    The class provides functions to query onto an index 2D Cartesian grid and maintains a mapping between space coordinates defined
    by polygons and the index into the polygon array.

    Custom regions can be easily created by using the from_polygon classmethod. This function will accept an arbitrary closed
    polygon and return a CartesianGrid class with only points inside the polygon to be valid.
    """
    def __init__(self, polygons, dh, name='cartesian2d', mask=None):
        self.polygons = polygons
        self.poly_mask = mask
        self.dh = dh
        self.name = name
        a, xs, ys = self._build_bitmask_vec()
        # in mask, True = bad value and False = good value
        self.bbox_mask = a[:,:,0]
        # contains the mapping from polygon_index to the mask
        self.idx_map = a[:,:,1]
        # index values of polygons array into the 2d cartesian grid, based on the midpoint.
        self.xs = xs
        self.ys = ys
        #Bounds [origin, top_right]
        orgs = self.origins()
        self.bounds = numpy.column_stack((orgs,orgs+dh))

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    @property
    def num_nodes(self):
        """ Number of polygons in region """
        return len(self.polygons)

    def get_index_of(self, lons, lats):
        """ Returns the index of lons, lats in self.polygons

        Args:
            lons: ndarray-like
            lats: ndarray-like

        Returns:
            idx: ndarray-like
        """
        idx = bin1d_vec(numpy.array(lons), self.xs)
        idy = bin1d_vec(numpy.array(lats), self.ys)
        if numpy.any(idx == -1) or numpy.any(idy == -1):
            raise ValueError("at least one lon and lat pair contain values that are outside of the valid region.")
        if numpy.any(self.bbox_mask[idy, idx] == 1):
            raise ValueError("at least one lon and lat pair contain values that are outside of the valid region.")
        return self.idx_map[idy,idx].astype(numpy.int)

    def get_location_of(self, indices):
        """
        Returns the polygon associated with the index idx.

        Args:
            idx: index of polygon in region

        Returns:
            Polygon

        """
        indices = list(indices)
        polys = [self.polygons[idx] for idx in indices]
        return polys

    def get_masked(self, lons, lats):
        """Returns bool array lons and lats are not included in the spatial region.

        .. note:: The ordering of lons and lats should correspond to the ordering of the lons and lats in the data.

        Args:
            lons: array-like
            lats: array-like

        Returns:
            idx: array-like
        """

        idx = bin1d_vec(lons, self.xs)
        idy = bin1d_vec(lats, self.ys)
        # handles the case where values are outside of the region
        bad_idx = numpy.where((idx == -1) | (idy == -1))
        mask = self.bbox_mask[idy, idx].astype(bool)
        # manually set values outside region
        mask[bad_idx] = True
        return mask

    def get_cartesian(self, data):
        """Returns 2d ndrray representation of the data set, corresponding to the bounding box.

        Args:
            data:
        """
        assert len(data) == len(self.polygons)
        results = numpy.zeros(self.bbox_mask.shape[:2])
        ny = len(self.ys)
        nx = len(self.xs)
        for i in range(ny):
            for j in range(nx):
                if self.bbox_mask[i, j] == 0:
                    idx = int(self.idx_map[i, j])
                    results[i, j] = data[idx]
                else:
                    results[i, j] = numpy.nan
        return results

    def get_bbox(self):
        """ Returns rectangular bounding box around region. """
        return (self.xs.min(), self.xs.max()+self.dh, self.ys.min(), self.ys.max()+self.dh)

    def midpoints(self):
        """ Returns midpoints of rectangular polygons in region """
        return numpy.array([poly.centroid() for poly in self.polygons])

    def origins(self):
        """ Returns origins of rectangular polygons in region """
        return numpy.array([poly.origin for poly in self.polygons])

    def to_dict(self):
        adict = {
            'name': str(self.name),
            'dh': float(self.dh),
            'polygons': [{'lat': float(poly.origin[1]), 'lon': float(poly.origin[0])} for poly in self.polygons],
            'class_id': self.__class__.__name__
        }
        return adict

    @classmethod
    def from_dict(cls, adict):
        """ Creates a region object from a dictionary """
        origins = adict.get('polygons', None)
        dh = adict.get('dh', None)
        magnitudes = adict.get('magnitudes', None)
        name = adict.get('name', 'CartesianGrid2D')

        if origins is None:
            raise AttributeError("cannot create region object without origins")
        if dh is None:
            raise AttributeError("cannot create region without dh")
        if origins is not None:
            try:
                origins = numpy.array([[adict['lon'], adict['lat']] for adict in origins])
            except:
                raise TypeError('origins must be numpy array like.')
        if magnitudes is not None:
            try:
                magnitudes = numpy.array(magnitudes)
            except:
                raise TypeError('magnitudes must be numpy array like.')

        out = cls.from_origins(origins, dh=dh, magnitudes=magnitudes, name=name)
        return out

    @classmethod
    def from_origins(cls, origins, dh=None, magnitudes=None, name=None):
        """Creates instance of class from 2d numpy.array of lon/lat origins.

        Note: Grid spacing should be constant in the entire region. This condition is not explicitly checked for for performance
        reasons.

        Args:
            origins (numpy.ndarray like): [:,0] = lons and [:,1] = lats
            magnitudes (numpy.array like): optional, if provided will bind magnitude information to the class.

        Returns:
            cls
        """
        # ensure we can access the lons and lats
        try:
            lons = origins[:,0]
            lats = origins[:,1]
        except (TypeError):
            raise TypeError("origins must be of type numpy.array or be numpy array like.")

        # dh must be regular, no explicit checking.
        if dh is None:
            dh2 = numpy.abs(lons[1]-lons[0])
            dh1 = numpy.abs(lats[1]-lats[0])
            dh = numpy.max([dh1, dh2])

        region = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(origins, dh)], dh, name=name)
        if magnitudes is not None:
            region.magnitudes = magnitudes
        return region

    def _build_bitmask_vec(self):
        """
        same as build mask but using vectorized calls to bin1d
        """
        # build bounding box of set of polygons based on origins
        nd_origins = numpy.array([poly.origin for poly in self.polygons])
        bbox = [(numpy.min(nd_origins[:, 0]), numpy.min(nd_origins[:, 1])),
                (numpy.max(nd_origins[:, 0]), numpy.max(nd_origins[:, 1]))]

        # get midpoints for hashing
        midpoints = numpy.array([poly.centroid() for poly in self.polygons])

        # set up grid over bounding box
        xs = cleaner_range(bbox[0][0], bbox[1][0], self.dh)
        ys = cleaner_range(bbox[0][1], bbox[1][1], self.dh)

        # set up mask array, 1 is index 0 is mask
        a = numpy.ones([len(ys), len(xs), 2])

        # set all indices to nan
        a[:, :, 1] = numpy.nan

        # bin1d returns the index of polygon within the cartesian grid
        idx = bin1d_vec(midpoints[:, 0], xs)
        idy = bin1d_vec(midpoints[:, 1], ys)

        for i in range(len(self.polygons)):
            a[idy[i], idx[i], 1] = int(i)
            # build mask in dim=0; here masked values are 1. see note below.
            if idx[i] >= 0 and idy[i] >= 0:
                if self.poly_mask is not None:
                    # note: csep1 gridded forecast file format convention states that a "1" indicates a valid cell, which is the opposite
                    # of the masking criterion
                    if self.poly_mask[i] == 1:
                        a[idy[i], idx[i], 0] = 0
                else:
                    a[idy[i], idx[i], 0] = 0

        return a, xs, ys

    def tight_bbox(self):
        # creates tight bounding box around the region, probably a faster way to do this.
        ny, nx = self.idx_map.shape
        asc = []
        desc = []
        for j in range(ny):
            row = self.idx_map[j, :]
            argmin = first_nonnan(row)
            argmax = last_nonnan(row)
            # points are stored clockwise
            poly_min = self.polygons[int(row[argmin])].points
            asc.insert(0, poly_min[0])
            asc.insert(0, poly_min[1])
            poly_max = self.polygons[int(row[argmax])].points
            lat_0 = poly_max[2][1]
            lat_1 = poly_max[3][1]
            # last two points are 'right hand side of polygon'
            if lat_0 < lat_1:
                desc.append(poly_max[2])
                desc.append(poly_max[3])
            else:
                desc.append(poly_max[3])
                desc.append(poly_max[2])
        # close the loop
        poly = np.array(asc + desc)
        sorted_idx = np.sort(np.unique(poly, return_index=True, axis=0)[1], kind='stable')
        unique_poly = poly[sorted_idx]
        unique_poly = np.append(unique_poly, [unique_poly[0, :]], axis=0)
        return unique_poly

    def get_cell_area(self):
        """ Compute the area of each polygon in sq. kilometers.

            Returns:
                out (numpy.array): numpy array containing cell area in km^2
        """
        area = numpy.zeros(self.num_nodes)
        for idx, origin in enumerate(self.origins()):
            top_right = origin + self.dh
            area[idx] = geographical_area_from_bounds(origin[0], origin[1], top_right[0], top_right[1])
        return area


def geographical_area_from_bounds(lon1, lat1, lon2, lat2):
    """
    Computes area of spatial cell identified by origin coordinate and top right cooridnate.
    The functions computes area only for square/rectangle bounding box by based on spherical earth assumption.
    Args:
        lon1,lat1 : Origin coordinates
        lon2,lat2: Top right coordinates
    Returns:
        Area of cell in Km2
    """
    if lon1 == lon2 or lat1 == lat2:
        #        print('Its line')
        return 0
    else:
        earth_radius_km = 6371.
        R2 = earth_radius_km ** 2
        rad_per_deg = numpy.pi / 180.0e0

        strip_area_steradian = 2 * numpy.pi * (1.0e0 - numpy.cos((90.0e0 - lat1) * rad_per_deg)) \
                           - 2 * numpy.pi * (1.0e0 - numpy.cos((90.0e0 - lat2) * rad_per_deg))
        area_km2 = strip_area_steradian * R2 / (360.0 / (lon2 - lon1))
        return area_km2

def quadtree_grid_bounds(quadk):
    """
    Computes the bottom-left and top-right coordinates corresponding to every quadkey

    Args:
        qk : Array of Strings
        Quadkeys.

    Returns:
        grid_coords : Array of floats
                    [lon1,lat1,lon2,lat2]

    """

    origin_lat = []
    origin_lon = []
    top_right_lon = []
    top_right_lat = []

    for i in range(len(quadk)):
        origin_lon.append(mercantile.bounds(mercantile.quadkey_to_tile(quadk[i])).west)
        origin_lat.append(mercantile.bounds(mercantile.quadkey_to_tile(quadk[i])).south)

        top_right_lon.append(mercantile.bounds(mercantile.quadkey_to_tile(quadk[i])).east)
        top_right_lat.append(mercantile.bounds(mercantile.quadkey_to_tile(quadk[i])).north)

    grid_origin = numpy.column_stack((numpy.array(origin_lon), numpy.array(origin_lat)))
    grid_top_right = numpy.column_stack((numpy.array(top_right_lon), numpy.array(top_right_lat)))
    grid_bounds = numpy.column_stack((grid_origin, grid_top_right))

    return grid_bounds

def compute_vertex_bounds(bound_point, tol=numpy.finfo(float).eps):
    """
    Wrapper function to compute vertices using bounding points for multiple points. Default tolerance is set to machine precision
    of floating point number.

    Args:
        bounding points: nx4 ndarray
                        [lon_origin, lat_origin, lon_top_right, lat_origin]
    Notes:
        (x,y) should be accessible like:
        #>>> origin coords = origin_points[:,0:1]
        #>>> Top right coords = origin_points[:,2:3]
    """
    bbox = ((bound_point[0], bound_point[1]),
            (bound_point[0], bound_point[3] - tol),
            (bound_point[2] - tol, bound_point[3] - tol),
            (bound_point[2] - tol, bound_point[1]))
    return bbox

def compute_vertices_bounds(bounds, tol=numpy.finfo(float).eps):
    """
    Wrapper function to compute vertices using bounding points for multiple points. Default tolerance is set to machine precision
    of floating point number.

    Args:
        bounding points: nx4 ndarray
                        [lon_origin, lat_origin, lon_top_right, lat_origin]
    Notes:
        (x,y) should be accessible like:
        #>>> origin coords = origin_points[:,0:1]
        #>>> Top right coords = origin_points[:,2:3]
    """
    return list(map(lambda x: compute_vertex_bounds(x, tol=tol), bounds))

def _create_tile(quadk, threshold, zoom, lon, lat, qk, num):
    """
    **Alert: This Function uses GLOBAL variable (qk) and (num).

        Provides multi-resolution quadtree spatial grid based on seismic density. It takes in a starting quadtree Tile (Quadkey),
        then keeps on increasing the zoom-level of every Tile (or dividing cell) recursively, unless every cell meets the cell dividion criteria.

        The primary criterion of dividing a parent cell into 4 child cells is a threshold on seismic denisity.
        The cells are divided unless evevry cell cas number of earthquakes less than "threshold".
        The cell division of any also stops if it reaches maximum zoom-level (zoom)

        Args:
            quadk : String
                    0, 1, 2, 3 or any desired starting level of Quad key.
            threshold : int
                    Max number of earthquakes/cell allowed
            zoom: int
                    Maximum zoom level allowed for a quadkey
            lon : float
                    longitudes of earthquakes in catalog
            lat : float
                    latitude of earthquakes in catalog

        Returns:
    """
    boundary = mercantile.bounds(mercantile.quadkey_to_tile(quadk))
    eqs = numpy.logical_and(numpy.logical_and(lon >= boundary.west, lat >= boundary.south),
                            numpy.logical_and(lon < boundary.east, lat < boundary.north))
    num_eqs = numpy.size(lat[eqs])
    #    global qk
    #    global num

    # Setting the Min Threshold of Area 1 sq. km. Instead of Depth

    if num_eqs > threshold and len(quadk) < zoom:  # #qk_area_km(quadk)>4:
        # print('inside If, Current Quad key ', quadk)
        # print('Length of Quadkey ', len(quadk))
        # # print('Num of Eqs ', num_eqs)

        _create_tile(quadk + '0', threshold, zoom, lon, lat, qk, num)

        _create_tile(quadk + '1', threshold, zoom, lon, lat, qk, num)

        _create_tile(quadk + '2', threshold, zoom, lon, lat, qk, num)

        _create_tile(quadk + '3', threshold, zoom, lon, lat, qk, num)

    else:
        # print('inside ELSE, Current Quad key ', quadk)
        # print('Num of Eqs ', num_eqs)
        #           qk = numpy.append(qk, quadk)
        qk.append(quadk)
        #            num = numpy.append(num, num_eqs)
        num.append(num_eqs)

def _create_tile_fix_len(quadk, zoom, qk):
    """
    ***Alert: This Function uses GLOBAL variable (qk).

        Provides single-resolution quadtree grid. It takes in a starting quadkey (or Quadrant of Globe),
        then keeps on keeps on dividing it into 4 children unless the maximum zoom-level is achieved
        Parameters
        ----------
        quadk : String
            0, 1, 2, 3 or any desired starting level of Quad key.
            zoom : TYPE
            Length of Quad Key OR Depth of grid.


            Returns
            -------
            None.
        """

    if len(quadk) < zoom:
        #        print('inside If, Current Quad key ', quadk)
        #        print('Len of QK: ', len(quadk))

        _create_tile_fix_len(quadk + '0', zoom, qk)

        _create_tile_fix_len(quadk + '1', zoom, qk)

        _create_tile_fix_len(quadk + '2', zoom, qk)

        _create_tile_fix_len(quadk + '3', zoom, qk)

    else:
        # print('inside ELSE, Current Quad key ', quadk)
        # print('Num of Eqs ', num_eqs)
        #        qk = numpy.append(qk, quadk)
        qk.append(quadk)

class QuadtreeGrid2D:
    """
    Respresents a 2D quadtree gridded region. The class provides functionality to generate multi-resolution or single-resolution quadtree grid.
    It also enables users to load already available quadtree grird. It also provides functions to query onto an index 2D grid ad maintains mapping
    between space coordinates and defined polygons and the index into the polygon array.

    Note: It is replica of CartesianGrid2D class but with quadtree approach, with implementation of all the relevant functions required to CSEP1 tests

    """

    def __init__(self, polygons, quadkeys, bounds, name='QuadtreeGrid2d', mask=None):
        """
        Args:
            polygons: Represents the object of class "polygons" defined through a collection of vertices.
                        This polygon is 2d and vertices are obtained as corner points of quadtree tile.
            quadkeys: Unique identifier of each quadtree tile. Quadkeys of every tile defines a grid cell.
                        This is the first thing computed while acquiring quadtree grid. Rest can be computed from this.
            bounds: number of cells x [lon1, lat1, lon2, lat2], corresponding to origin coordinates and top right coordinates fo each grid cell
            name: Name of grid
            mask: Masked cells. NotImplemented yet. Always keep it none
        """
        self.polygons = polygons
        self.quadkeys = quadkeys
        self.bounds = bounds
        self.cell_area = []
        self.poly_mask = mask
        self.name = name
        # a, xs, ys = self._get_idx_map_xs_ys()
        # self.xs = xs
        # self.ys = ys
        # self.idx_map = a
        # self.dh = 0.5 #Temporary use, until 'dh' is removed from plot_spatial_datasets() of forecast.plot.

    @property
    def num_nodes(self):
        """ Number of polygons in region """
        return len(self.polygons)

    def get_cell_area(self):
        """
        Calls function geographical_area_from_bounds and computes area of each grid cell. It also modified class variable "self.cell_area"
        It iterates over all the cells of grid and passes bounding coordinates of every cell to function  geographical_area_from_bounds
        """
        cell_area = numpy.array([geographical_area_from_bounds(bb[0],bb[1],bb[2],bb[3]) for bb in self.bounds])
        self.cell_area = cell_area
        return self.cell_area

    def get_index_of(self, lons, lats): 
        """ Returns the index of lons, lats in self.polygons

        Args:
            lons: ndarray-like
            lats: ndarray-like

        Returns:
            idx: ndarray-like
        """
        # If its array or many coords
        if isinstance(lons, (list, numpy.ndarray)):
            idx = []
            for i in range(len(lons)):
                idx = numpy.append(idx, self._find_location(lons[i], lats[i]))
            idx = idx.astype(int)
            return idx
        # It its just one Lon/Lon
        if isinstance(lons, (int, float)):
            idx = self._find_location(lons, lats)
            return idx
        return None

    def _find_location(self, lon, lat):
        """ Takes in single Lon and Lat and finds its Polygon Index.

        Returns:
            index number of polyons
        """
        loc = numpy.logical_and(numpy.logical_and(lon >= self.bounds[:, 0], lat >= self.bounds[:, 1]),
                                    numpy.logical_and(lon < self.bounds[:, 2], lat < self.bounds[:, 3]))
        if len(numpy.where(loc == True)[0]) > 0:
            return numpy.where(loc == True)[0][0]
        else:
            return numpy.where(loc == True)[0]

    def get_location_of(self, indices):
        """ Returns the polygon associated with the index idx.

        Args:
            idx: index of polygon in region

        Returns:
            Polygon
        """
        indices = list(indices)
        polys = [self.polygons[idx] for idx in indices]
        return polys

    def _get_spatial_counts(self, catalog, mag_bins=None):
        """ Gets the number of earthquakes in each cell for available catalog.
        Uses QuadtreeGrid2D.get_index_of function to map every earthquake location to its corresponding cell

        Args:
            catalog: CSEP Catalog
            mag_bins: Magnitude discritization used in earthquake forecast mdoel
                      Note: mag_bins are only required to filter catalog for minimum magnitude

        Return:
            spatial counts: Number of earthquakes in each cell

        """
        if mag_bins is None or mag_bins == []:
            mag_bins = catalog.magnitudes

        if min(catalog.get_magnitudes()) < min(mag_bins):
            print("-----Warning-----")
            print("Catalog contains magnitudes below the min magnitude range")
            print("Filtering catalog with Magnitude: ", min(mag_bins))
            catalog.filter('magnitude >= ' + str(min(mag_bins)))

        if min(catalog.get_latitudes()) < self.get_bbox()[2] or max(catalog.get_latitudes()) > self.get_bbox()[3]:
            print("----Warning---")
            print("Catalog exceeds grid bounds, so catalog filtering")
            catalog.filter('latitude < ' + str(self.get_bbox()[3]))
            catalog.filter('latitude > ' + str(self.get_bbox()[2]))

        lon = catalog.get_longitudes()
        lat = catalog.get_latitudes()

        out = numpy.zeros(len(self.quadkeys))
        idx = self.get_index_of(lon, lat)
        numpy.add.at(out, idx, 1)

        return out

    def _get_spatial_magnitude_counts(self, catalog, mag_bins=None):
        """
        Gets the number of earthquakes in for each spatio-magnitude bin for available catalog
        Uses QuadtreeGrid2D.get_index_of function to map every earthquake location to its corresponding cell
        Uses bin1d_vec function to map earthquake magnitude to its respecrtive bin.

        Args:
            catalog: CSEPCatalog
            mag_bins: Magnitude discritization used in earthquake forecast model

        Return:
            Spatial-magnitude counts

        """
        if mag_bins is None or mag_bins == []:
            mag_bins = catalog.magnitudes

        if min(catalog.get_magnitudes()) < min(mag_bins):
            print("-----Warning-----")
            print("Catalog contains magnitudes below the min magnitude range")
            print("Filtering catalog with Magnitude: ", min(mag_bins))
            catalog.filter('magnitude >= ' + str(min(mag_bins)))

        if min(catalog.get_latitudes()) < self.get_bbox()[2] or max(catalog.get_latitudes()) > self.get_bbox()[3]:
            print("----Warning---")
            print("Catalog exceeds grid bounds filtering events outside of the region boundary")
            catalog.filter('latitude < ' + str(self.get_bbox()[3]))
            catalog.filter('latitude > ' + str(self.get_bbox()[2]))

        lon = catalog.get_longitudes()
        lat = catalog.get_latitudes()
        mag = catalog.get_magnitudes()
        out = numpy.zeros([len(self.quadkeys), len(mag_bins)])

        idx_loc = self.get_index_of(lon, lat)
        idx_mag = bin1d_vec(mag, mag_bins, tol=0.00001, right_continuous=True)

        numpy.add.at(out, (idx_loc, idx_mag), 1)

        return out

    def get_bbox(self):
        """ Returns rectangular bounding box around region. """
        #        return (self.xs.min(), self.xs.max(), self.ys.min(), self.ys.max())
        return (min(self.bounds[:, 0]), max(self.bounds[:, 2]), min(self.bounds[:, 1]), max(self.bounds[:, 3]))

    def midpoints(self):
        """ Returns midpoints of rectangular polygons in region """
        return numpy.array([poly.centroid() for poly in self.polygons])

    def origins(self):
        """ Returns origins of rectangular polygons in region """
        return numpy.array([poly.origin for poly in self.polygons])

    def to_dict(self):
        adict = {
            'name': str(self.name),
            'polygons': [{'lat': float(poly.origin[1]), 'lon': float(poly.origin[0])} for poly in self.polygons]
        }
        return adict

    def save_quadtree(self, filename):
        """ Saves the quadtree grid (quadkeys) in a text file

            Args:
                filename (str): filename to store file
        """
        numpy.savetxt(filename, self.quadkeys, delimiter=',', fmt='%s')

    @classmethod
    def from_catalog(cls, catalog, threshold, zoom=11, magnitudes=None, name=None):
        """
        Creates instance of class from 2d numpy.array of lon/lat of Catalog.
        Provides multi-resolution quadtree spatial grid based on seismic density. It starts from whole globe as 4 cells (Quadkeys:'0','1','2','3'),
        then keeps on increasing the zoom-level of every Tile recursively, unless every cell meets the division criteria.

        The primary criterion of dividing a parent cell into 4 child cells is a threshold on seismic density.
        The cells are divided unless every cell has number of earthquakes less than "threshold".
        The division of a cell also stops if it reaches maximum zoom-level (zoom)

        Args:
            catalog (CSEPCatalog): catalog used to create quadtree
            threshold (int): Max earthquakes allowed per cells
            zoom (int): Max zoom allowed for a cell
            magnitudes (array-like): left end values of magnitude discretization

        Returns:
            instance of QuadtreeGrid2D
        """

        lon = catalog.get_longitudes()
        lat = catalog.get_latitudes()

        qk = []
        num = []

        _create_tile('0', threshold, zoom, lon, lat, qk, num)
        _create_tile('1', threshold, zoom, lon, lat, qk, num)
        _create_tile('2', threshold, zoom, lon, lat, qk, num)
        _create_tile('3', threshold, zoom, lon, lat, qk, num)

        qk = numpy.array(qk)
        bounds = quadtree_grid_bounds(qk)
        region = QuadtreeGrid2D(
            [Polygon(bbox) for bbox in compute_vertices_bounds(bounds)],
            qk,
            bounds,
            name=name)

        if magnitudes is not None:
            region.magnitudes = magnitudes

        return region

    @classmethod
    def from_single_resolution(cls, zoom, magnitudes=None, name=None):
        """ Creates instance of class at single-resolution using provided zoom-level.
        Provides single-resolution quadtree grid. It starts from whole globe as 4 cells (Quadkeys:'0','1','2','3'),
        then keeps on keeps on dividing every cell into 4 children unless the maximum zoom-level is achieved

        Args:
            zoom: Max zoom allowed for a cell
            magnitude: magnitude discretization

        Returns:
            instance of QuadtreeGrid2D
        """

        qk = []
        _create_tile_fix_len('0', zoom, qk)
        _create_tile_fix_len('1', zoom, qk)
        _create_tile_fix_len('2', zoom, qk)
        _create_tile_fix_len('3', zoom, qk)

        qk = numpy.array(qk)

        bounds = quadtree_grid_bounds(qk)

        region = QuadtreeGrid2D([Polygon(bbox) for bbox in compute_vertices_bounds(bounds)], qk, bounds,
                                name=name)

        if magnitudes is not None:
            region.magnitudes = magnitudes
        return region

    @classmethod
    def from_quadkeys(cls, quadk, magnitudes=None, name=None):
        """ Creates instance of class from available quadtree grid.

        Args:
            quadk (list): List of quad keys strings corresponding to an already available quadtree grid
            magnitudes (array-like): left end-points of magnitude discretization

        Returns:
            instance of QuadtreeGrid2D
        """
        bounds = quadtree_grid_bounds(numpy.array(quadk))

        region = QuadtreeGrid2D([Polygon(bbox) for bbox in compute_vertices_bounds(bounds)], quadk, bounds,
                                name=name)

        if magnitudes is not None:
            region.magnitudes = magnitudes

        return region

    def _get_idx_map_xs_ys(self):
        print('inside _get_idx_map')
        nd_origins = numpy.array([poly.origin for poly in self.polygons])
        xs = numpy.unique(nd_origins[:, 0])
        ys = numpy.unique(nd_origins[:, 1])
        ny = len(ys)
        nx = len(xs)
        #Get the index map
        a = numpy.zeros([ny, nx])
        for i in range(nx):
            for j in range(ny):
                idx = self.get_index_of(xs[i], ys[j])
                a[j, i] = idx
        return a, xs, ys

    def get_cartesian(self, data):
        """ Returns 2d ndrray representation of the data set, corresponding to the bounding box.

        Args:
            data (numpy.array): array of values corresponding to cells in the quadtree region

        Returns:
            results (numpy.array): 2d numpy array with rates on cartesian grid

        """
        a, xs, ys = self._get_idx_map_xs_ys()
        self.xs = xs
        self.ys = ys
        self.idx_map = a
        assert len(data) == len(self.polygons)
        ny = len(self.ys)
        nx = len(self.xs)
        results = numpy.zeros([ny, nx])
        for i in range(nx):
            for j in range(ny):
                idx = int(self.idx_map[j,i])
                results[j, i] = data[idx]
        return results

    def tight_bbox(self):
        # creates tight bounding box around the region, probably a faster way to do this.
        ny, nx = self.idx_map.shape
        asc = []
        desc = []
        for j in range(ny):
            row = self.idx_map[j, :]
            argmin = first_nonnan(row)
            argmax = last_nonnan(row)
            # points are stored clockwise
            poly_min = self.polygons[int(row[argmin])].points
            asc.insert(0, poly_min[0])
            asc.insert(0, poly_min[1])
            poly_max = self.polygons[int(row[argmax])].points
            lat_0 = poly_max[2][1]
            lat_1 = poly_max[3][1]
            # last two points are 'right hand side of polygon'
            if lat_0 < lat_1:
                desc.append(poly_max[2])
                desc.append(poly_max[3])
            else:
                desc.append(poly_max[3])
                desc.append(poly_max[2])
        # close the loop
        poly = np.array(asc + desc)
        sorted_idx = np.sort(np.unique(poly, return_index=True, axis=0)[1], kind='stable')
        unique_poly = poly[sorted_idx]
        unique_poly = np.append(unique_poly, [unique_poly[0, :]], axis=0)
        return unique_poly


def california_quadtree_region(magnitudes=None, name="california-quadtree"):
    """
    Returns object of QuadtreeGrid2D representing quadtree grid for California RELM testing region.
    The grid is already generated at zoom-level = 12 and it is loaded through classmethod: QuadtreeGrid2D.from_quadkeys
    The grid cells at zoom level 12 are selected using the external boundary of RELM california region.
    This grid can be used to create gridded datasets for earthquake forecasts.


    Args:
        magnitudes: Magnitude discretization
        name: string

    Returns:
        :class:`csep.core.spatial.QuadtreeGrid2D

    """
    # use default file path from python package
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'california_qk_zoom=12.txt')
    qk = numpy.genfromtxt(filepath, delimiter=',', dtype='str')
    california_region = QuadtreeGrid2D.from_quadkeys(qk, magnitudes=magnitudes, name=name)
    return california_region

#--------------- Forecast mapping from one grid to another ----------
def geographical_area_from_qk(quadk):
    """
    Wrapper around function geographical_area_from_bounds
    """
    bounds = tile_bounds(quadk)
    return geographical_area_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3])


def tile_bounds(quad_cell_id):
    """
    It takes in a single Quadkkey and returns lat,longs of two diagonal corners using mercantile
    Parameters
    ----------
    quad_cell_id : Stirng
        Quad key of a cell.

    Returns
    -------
    bounds : Mercantile object
        Latitude and Longitude of bottom left AND top right corners.

    """
    # t = Tile.from_quad_tree('{}'.format(quad_cell_id)) # tile from a quad tree repr string #This will change for Mercantile
    # bounds = t.bounds
    bounds = mercantile.bounds(mercantile.quadkey_to_tile(quad_cell_id))
    return [bounds.west, bounds.south, bounds.east, bounds.north]


def create_polygon(fg):
    """
    Required for parallel processing
    """
    return shapely.geometry.Polygon([(fg[0], fg[1]), (fg[2], fg[1]), (fg[2], fg[3]), (fg[0], fg[3])])


def calc_cell_area(cell):
    """
    Required for parallel processing
    """
    return geographical_area_from_bounds(cell[0], cell[1], cell[2], cell[3])


def _map_overlapping_cells(fcst_grid_poly, fcst_cell_area, fcst_rate_poly, target_poly):  # ,
    """
    ------Computationally Expensive --- To be used only for Cells that do not directly conside with target poly -----
    Note: ALERT !!! This function uses 3 global variables, i.e. fcst_grid_poly, fcst_cell_area, fcst_rate_poly

    This function takes 1 target polygon, upon which forecasts are to be mapped. Finds all the cells of forecast grid that
    match with this polygon and then maps the forecast rate of those cells according to area.

    fcst_grid_polygon (Global variable in memory): The grid that needs to be mapped on target_poly
    fcst_rate_poly (Global variable in memory): The forecast that needs to be mapped on target grid polygon
    fcst_cell_area (Global variable in memory): The cell area of forecast grid

    Input:
        target_poly: One polygon upon which forecast grid is to be mapped.
    returns:
        The forecast rate received by target_poly
    """
    map_rate = numpy.array([0])  # --ithy masla ae.
    #    map_rate = numpy.zeros(len(fcst_rate_poly))

    for j in range(len(fcst_grid_poly)):
        # Iterates over ALL the cells of Forecast grid and find the cells that overlap with target cell (poly).
        if target_poly.intersects(fcst_grid_poly[j]):  # overlaps
            #           print('L5 Cell:',j, '--> L6Quadtree:')
            intersect = target_poly.intersection(fcst_grid_poly[j])
            shared_area = geographical_area_from_bounds(intersect.bounds[0], intersect.bounds[1], intersect.bounds[2],
                                                        intersect.bounds[3])
            map_rate = map_rate + (fcst_rate_poly[j] * (shared_area / fcst_cell_area[j]))
    return map_rate


def _map_exact_inside_cells(fcst_grid, fcst_rate, boundary):
    """
    Note: ALTER !!! Uses 2 Global variables. fcst_grid, fcst_rate

    Takes a cell_boundary and finds all those fcst_grid cells that fit exactly inside of it
    And then sum-up the rates of all those cells fitting inside it to get forecast rate for boundary_cell

    Inputs:
        boundary: 1 cell with [lon1, lat1, lon2, lat2]
    returns:
        1 - sum of forecast_rates for cell that fall totally inside of boundary cell
        2 - Array of the corresponding cells that fall inside
    """
    c = numpy.logical_and(numpy.logical_and(fcst_grid[:, 0] >= boundary[0], fcst_grid[:, 1] >= boundary[1]),
                          numpy.logical_and(fcst_grid[:, 2] <= boundary[2], fcst_grid[:, 3] <= boundary[3]))

    exact_cells = numpy.where(c == True)

    #    return sum(fcst_rate[c]), exact_cells
    return numpy.sum(fcst_rate[c], axis=0), exact_cells


def forecast_mapping_generic(target_grid, fcst_grid, fcst_rate, ncpu=None):
    """
    ----Both Aggregation and De-Aggregation ----
    ***It is a wrapper function that uses 4 functions in respective order
    i.e. _map_exact_cells, _map_overlapping_cells, calc_cell_area, create_polygon***

    Maps the forecast rates of one grid to another grid using parallel processing
    Works in two steps:
        1 - Maps all those cells that fall entirely on target cells
        2 - The cells that overlap with multiple cells, map them according to cell area
    Inputs:
        target_grid: Target grid bounds, upon which forecast is to be mapped.
                        [n x 4] array, Bottom left and Top Right corners
                        [lon1, lat1, lon2, lat2]
        fcst_grid: Available grid that is available with forecast
                            Same as bounds_targets
        fcst_rate: Forecast rates to be mapped.
                    [n x mbins]

    Returns:
        target_rates:
                Forecast rates mapped on the target grid
                [nx1]
    """
    print(len(fcst_grid))
    print('--First Step: Exact Cell mapping--')
    #    tstart = time.time()
    #    t1 = time.time()
    if ncpu==None:
        ncpu = mp.cpu_count()
        pool = mp.Pool(ncpu)
    else:
        pool = mp.Pool(ncpu)  # mp.cpu_count()
    print('Number of CPUs :',ncpu)

    func_exact = partial(_map_exact_inside_cells, fcst_grid, fcst_rate)
    exact_rate = pool.map(func_exact, [poly for poly in target_grid])
    pool.close()
    #    t2 = time.time()
    #    print('Time taken for First Step :', t2-t1)

    exact_cells = []
    exact_rate_tgt = []
    for i in range(len(exact_rate)):
        exact_cells.append(exact_rate[i][1][0])
        exact_rate_tgt.append(exact_rate[i][0])

    exact_cells = numpy.concatenate(exact_cells)
    print('Number of Exact Cells: ', len(exact_cells))
    # If Len(exact_cell) == len(fcst_grid):
    #        map_rate =  exact_rate_tgt
    # return map_rate
    # else:

    # Exclude all those cells from Grid that have already fallen entirely inside any cell of Target Grid
    fcst_rate_poly = numpy.delete(fcst_rate, exact_cells, axis=0)  # GLOBAL pARAMETER -Rates
    lft_fcst_grid = numpy.delete(fcst_grid, exact_cells, axis=0)

    # ---Lets play now only with those cells are overlapping with multiple target cells

    ##Get the polygon of Remaining Forecast grid Cells
    pool = mp.Pool(ncpu)
    fcst_grid_poly = pool.map(create_polygon, [i for i in lft_fcst_grid])  # GLOBAL pARAMETER -Grid
    pool.close()

    # Get the Cell Area of forecast grid
    pool = mp.Pool(ncpu)
    fcst_cell_area = pool.map(calc_cell_area, [i for i in lft_fcst_grid])  # GLOBAL pARAMETER  -Area
    pool.close()

    print('Calculate target polygons')
    pool = mp.Pool(ncpu)
    target_grid_poly = pool.map(create_polygon, [i for i in target_grid])
    pool.close()

    print('--2nd Step: Start Polygon mapping--')
    #    t1 = time.time()
    pool = mp.Pool(ncpu)  # mp.cpu_count()
    func_overlapping = partial(_map_overlapping_cells, fcst_grid_poly, fcst_cell_area, fcst_rate_poly)
    rate_tgt = pool.map(func_overlapping, [poly for poly in target_grid_poly])  # Uses above three Global Parameters
    pool.close()
    #    t2 = time.time()
    #    print('Time for Mapping: ', t2-t1)
    #    exact_rate_tgt = numpy.vstack(exact_rate_tgt)
    print('Shape of Exact cell :', numpy.shape(exact_rate_tgt))
    print('Shape of Shared Cells :', numpy.shape(rate_tgt))
    # Zero padding in shared rates, which dont receive any rates.

    zero_pad_len = numpy.shape(fcst_rate)[1]
    for i in range(len(rate_tgt)):
        if len(rate_tgt[i]) < zero_pad_len:
            print('Zero Padding in shared Rates :', i)
            rate_tgt[i] = numpy.zeros(zero_pad_len)

    map_rate = numpy.add(rate_tgt, exact_rate_tgt)
    print('Difference in Forecast Rate = ', numpy.sum(map_rate) - numpy.sum(fcst_rate))
    #    numpy.savetxt('forecast mapping/gear_rate_zoom='+str(zoom[z])+'.csv', map_rate, delimiter=',')
    #    tend= time.time()
    #    print('Time for forecast mapping: ',tend-tstart)
    return map_rate


#    return numpy.vstack(map_rate)
#    return exact_rate_tgt, rate_tgt

# -----

##---Function for Forecast De-Aggregation
def forecast_deaggregate_qk(qk_high_zoom, qk_low_zoom, forecast_low_zoom):
    """
    Forecast mapping from low Zoom grid (Big Cells) to High Zoom grid: De-Aggregation
    When each low zoom bin is divided into 04 cells, the forecast is area-wise divided in each cell

    Parameters
    ----------
    qk_high_zoom : Array of Strings
        Quadkeys with Low Threshold
    qk_low_zoom : Array of Strings
       Quadkeys with High Threshold.
    forecast_low_zoom : @D numpy array
        Forecast for bins with high threshold EQs

    Returns
    -------
    qk_cast : Array of Strings
        Quadkeys = High Zoom quadkeys
    forecast_cast : 2D numpy array
        High Zoom forecast obtained from dividing Low Zoom forecast

    """
    assert (len(qk_high_zoom) > len(qk_low_zoom)), "High Zoom Quadkey should be given first"
    print("Length of Low Zoom QK", len(qk_low_zoom))
    print("Length of Low ZOom Forecast", len(forecast_low_zoom))
    assert (len(qk_low_zoom) == len(forecast_low_zoom)), "Make Sure Low Zoom Quadkeys and Forecasts are consistent"

    #    qk_cast = numpy.array([])
    #    qk_cast = numpy.append(qk_cast,qk_low_zoom)
    qk_cast = list(qk_low_zoom)
    forecast_cast = list(forecast_low_zoom)

    i = 0

    while (i < len(qk_high_zoom)):
        if qk_cast[i] == qk_high_zoom[i]:
            #            if len(qk_cast) == i:
            #                qk_cast.append(qk_low_zoom[i])  #---NOW
            #                forecast_cast.append(forecast_low_zoom[i]) #--NOW
            i = i + 1
            # print("Into IF")
            # print('i = ', i)
        else:
            # Put Sanity check here for comparing previous values of strings.
            # print('into ELSE..')
            # print('i = ', i)
            #            tmpold = copy.copy(qk_cast[i])
            #            tmpf = copy.copy(forecast_cast[i])

            #            ----NOW

            # --Get the Area of Bigger Cell, before changing it ...
            qk_old = qk_cast[i]
            area = geographical_area_from_qk(qk_old)

            qk_cast[i] = qk_old + '0'
            qk_cast.insert(i + 1, qk_old + '1')
            qk_cast.insert(i + 2, qk_old + '2')
            qk_cast.insert(i + 3, qk_old + '3')

            # Get Area0 = tmpold+'0'
            area_0 = geographical_area_from_qk(qk_cast[i])
            area_1 = geographical_area_from_qk(qk_cast[i + 1])
            area_2 = geographical_area_from_qk(qk_cast[i + 2])
            area_3 = geographical_area_from_qk(qk_cast[i + 3])

            f_old = forecast_cast[i]
            forecast_cast[i] = f_old * (area_0 / area)
            forecast_cast.insert(i + 1, f_old * (area_1 / area))
            forecast_cast.insert(i + 2, f_old * (area_2 / area))
            forecast_cast.insert(i + 3, f_old * (area_3 / area))

    return  numpy.vstack(forecast_cast) #numpy.array(qk_cast),


def forecast_mapping(target_grid, forecast_gridded, only_deaggregate=False,  ncpu=None):
    """
    --Wrapper function over "forecat_mapping_generic" and forecast_deaggregate_qk"--
    Forecast mapping onto Target Grid
    It only De-aggregates using forecast_deaggregate_qk
    OR
    Uses generic function named as forecast_mapping_parallel

    target_grid: csep.core.region.CastesianGrid2D or QuadtreeGrid2D
    forecast_gridded: csep.core.forecast with other grid.
    only_de-aggregate: Flag (True or False)
        Note: set the flag "only_deagregate = True" Only if one is sure that both grids are Quadtree and
        Target grid is high-resolution at every level than the other grid.
    """
    from csep.core.forecasts import GriddedForecast
    if only_deaggregate:
        qk_target = target_grid.quadkeys
        qk = forecast_gridded.region.quadkeys
        data = forecast_gridded.data
        data_mapped_qk = forecast_deaggregate_qk(qk_target, qk, data)
        target_forecast = GriddedForecast(data=data_mapped_qk, region=target_grid,
                                          magnitudes=forecast_gridded.magnitudes)
    else:
        bounds_target = target_grid.bounds
        bounds = forecast_gridded.region.bounds
        data = forecast_gridded.data
        data_mapped_bounds = forecast_mapping_generic(bounds_target, bounds, data, ncpu=ncpu)
        # Using GriddedForecast, which I imported. Check for Circular imports?
        target_forecast = GriddedForecast(data=data_mapped_bounds, region=target_grid,
                                          magnitudes=forecast_gridded.magnitudes)
    return target_forecast
