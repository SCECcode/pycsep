# Python imports
import itertools
import os
from itertools import compress
from xml.etree import ElementTree as ET

# Third-party imports
import numpy
import numpy as np
import mercantile
from shapely import geometry
from shapely.ops import unary_union

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
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, magnitudes=magnitudes, name=name)

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
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh,magnitudes=magnitudes,
                                  name=name)

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
    italy_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh,
                                   magnitudes=magnitudes, name=name)

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
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, magnitudes=magnitudes,
                                  name=name)

    return relm_region

def nz_csep_region(dh_scale=1, magnitudes=None, name="csep-nz", use_midpoint=True):
    """ Return collection region for the New Zealand CSEP testing region

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
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'nz.testing.nodes.dat')
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
    nz_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, magnitudes=magnitudes,
                                name=name)

    return nz_region

def nz_csep_collection_region(dh_scale=1, magnitudes=None, name="csep-nz-collection", use_midpoint=True):
    """ Return collection region for the New Zealand CSEP collection region

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
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'nz.collection.nodes.dat')
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
    nz_collection_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh,
                                           magnitudes=magnitudes, name=name)

    return nz_collection_region

def global_region(dh=0.1, name="global", magnitudes=None):
    """ Creates a global region used for evaluating gridded forecasts on the global scale.


    Args:
        dh (float): Spacing for lat and lon discretization.
        name (str): Name of the region
        magnitudes(list/array): Array containing the magnitude bins

    Returns:
        csep.utils.CartesianGrid2D: Global region
    """
    # generate latitudes

    lons = cleaner_range(-180.0, 180.0, dh)[:-1]
    lats = cleaner_range(-90, 90.0, dh)[:-1]
    coords = itertools.product(lons,lats)
    region = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(coords, dh)], dh,
                             magnitudes=magnitudes, name=name)

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
    return cleaner_range(start_magnitude, end_magnitude, dmw)

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

def _bin_catalog_spatio_magnitude_counts(lons, lats, mags, n_poly, mask, idx_map, binx, biny, mag_bins, tol=None):
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
    def __init__(self, polygons, dh, name='cartesian2d', mask=None, magnitudes=None):
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
        # Bounds [origin, top_right]
        orgs = self.origins()
        self.bounds = numpy.column_stack((orgs, orgs + dh))
        self.magnitudes = magnitudes

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
        idx = bin1d_vec(lons, self.xs)
        idy = bin1d_vec(lats, self.ys)
        if numpy.any(idx == -1) or numpy.any(idy == -1):
            raise ValueError("at least one lon and lat pair contain values that are outside of the valid region.")
        if numpy.any(self.bbox_mask[idy, idx] == 1):
            raise ValueError("at least one lon and lat pair contain values that are outside of the valid region.")
        return self.idx_map[idy, idx].astype(numpy.int64)

    def get_location_of(self, indices):
        """
        Returns the polygon associated with the index idx.

        Args:
            indices: index of polygon in region

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
            data: An array of values, whose indices corresponds to each cell of the region

        Returns:
            A 2D array of values, corresponding to the original data projected onto the region.
            Values outside the region polygon are represented as np.nan
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

        region = CartesianGrid2D(polygons=[Polygon(bbox) for bbox in compute_vertices(origins, dh)],
                                 dh=dh,
                                 magnitudes=magnitudes,
                                 name=name)

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

    def tight_bbox(self, precision=4):
        # creates tight bounding box around the region
        poly = np.array([i.points for i in self.polygons])

        sorted_idx = np.sort(np.unique(poly, return_index=True, axis=0)[1], kind='stable')
        unique_poly = poly[sorted_idx]

        # merges all the cell polygons into one
        polygons = [geometry.Polygon(np.round(i, precision)) for i in unique_poly]
        joined_poly = unary_union(polygons)
        bounds = np.array([i for i in joined_poly.boundary.xy]).T

        return bounds

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
        idx_mag = bin1d_vec(mag, mag_bins, right_continuous=True)

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

