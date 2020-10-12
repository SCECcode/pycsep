# Python imports
import itertools
import os
from itertools import compress
from xml.etree import ElementTree as ET

# Third-party imports
import matplotlib.path
import numpy

# PyCSEP imports
import numpy as np
import pyproj

from csep.utils.calc import bin1d_vec
from csep.utils.scaling_relationships import WellsAndCoppersmith

def california_relm_collection_region(dh_scale=1, magnitudes=None, name="relm-california-collection"):
    """ Return collection region for California RELM testing region

        Args:
            dh_scale (int): factor of two multiple to change the grid size
            mangitudes (array-like): array representing the lower bin edges of the magnitude bins
            name (str): human readable identifer
    """
    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

    # we can hard-code the dh because we hard-code the filename
    dh = 0.1
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'RELMCollectionArea.dat')
    midpoints = numpy.loadtxt(filepath)
    origins = midpoints - dh / 2

    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        relm_region.magnitudes = magnitudes

    return relm_region

def california_relm_region(dh_scale=1, magnitudes=None, name="relm-california"):
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
    midpoints, dh = parse_csep_template(csep_template)
    origins = numpy.array(midpoints) - dh / 2

    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        relm_region.magnitudes = magnitudes

    return relm_region

def italy_csep_region(dh_scale=1, magnitudes=None, name="csep-italy"):
    """
        Returns class representing Italian testing region.

        This region can be used to create gridded datasets for earthquake forecasts. The region is defined by the
        file 'forecast.italy.M5.xml' and contains a spatially gridded region with 0.1° x 0.1° cells.

        Args:
            dh_scale: can resample this grid by factors of 2
            magnitudes (array-like): bin edges for magnitudes. if provided, will be bound to the output region class.
                                     this argument provides a short-cut for creating space-magnitude regions.

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
    midpoints, dh = parse_csep_template(csep_template)
    origins = numpy.array(midpoints) - dh / 2

    if dh_scale > 1:
        origins = increase_grid_resolution(origins, dh, dh_scale)
        dh = dh / dh_scale

    # turn points into polygons and make region object
    bboxes = compute_vertices(origins, dh)
    italy_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name=name)

    if magnitudes is not None:
        italy_region.magnitudes = magnitudes

    return italy_region

def italy_csep_collection_region(dh_scale=1, magnitudes=None, name="csep-italy-collection"):
    """ Return collection region for Italy CSEP collection region

        Args:
            dh_scale (int): factor of two multiple to change the grid size
            mangitudes (array-like): array representing the lower bin edges of the magnitude bins
            name (str): human readable identifer
    """
    if dh_scale % 2 != 0 and dh_scale != 1:
        raise ValueError("dh_scale must be a factor of two or dh_scale must equal unity.")

    # we can hard-code the dh because we hard-code the filename
    dh = 0.1
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(root_dir, 'artifacts', 'Regions', 'italy.collection.nodes.dat')
    midpoints = numpy.loadtxt(filepath)
    origins = midpoints - dh / 2

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
    lats = numpy.arange(-90.0, 89.9 + dh/2, dh)
    lons = numpy.arange(-180, 179.9 + dh/2, dh)
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
    return numpy.arange(start_magnitude, end_magnitude+dmw/2, dmw)

def create_space_magnitude_region(region, magnitudes):
    """Simple wrapper to create space-magnitude region """
    if not isinstance(region, CartesianGrid2D):
        raise TypeError("region must be CartesianGrid2D")
    # bind to region class
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
        >>> x_coords = origin_points[:,0]
        >>> y_coords = origin_points[:,1]

    """
    return list(map(lambda x: compute_vertex(x, dh, tol=tol), origin_points))

def _build_bitmask_vec(polygons, dh):
    """
    same as build mask but using vectorized calls to bin1d
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

    # set all indices to nan
    a[:,:,1] = numpy.nan

    # bin1d returns the index of polygon within the cartesian grid
    idx = bin1d_vec(midpoints[:, 0], xs)
    idy = bin1d_vec(midpoints[:, 1], ys)

    for i in range(len(polygons)):
        # store index of polygon in dim=1
        a[idy[i], idx[i], 1] = int(i)
        # build mask in dim=0; here we masked values are 1
        if idx[i] >= 0 and idy[i] >= 0:
            a[idy[i], idx[i], 0] = 0

    return a, xs, ys

def _bin_catalog_spatio_magnitude_counts(lons, lats, mags, n_poly, mask, idx_map, binx, biny, mag_bins):
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
    mag_idxs = bin1d_vec(mags, mag_bins, right_continuous=True)
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

class Polygon:
    """
    Represents polygons defined through a collection of vertices.

    This polygon is assumed to be 2d, but could contain an arbitrary number of vertices. The path is treated as not being
    closed.
    """
    def __init__(self, points):
        # instance members
        self.points = points
        self.origin = self.points[0]

        # https://matplotlib.org/3.1.1/api/path_api.html
        self.path = matplotlib.path.Path(self.points)

    def __str__(self):
        return str(self.origin)

    def contains(self, points):
        """ Returns a bool array which is True if the path contains the corresponding point.

        Args:
            points: 2d numpy array

        """
        nd_points = np.array(points)
        if nd_points.ndim == 1:
            nd_points = nd_points.reshape(1,-1)
        return self.path.contains_points(nd_points)

    def centroid(self):
        """ return the centroid of the polygon."""
        c0, c1 = 0, 0
        k = len(self.points)
        for p in self.points:
            c0 = c0 + p[0]
            c1 = c1 + p[1]
        return c0 / k, c1 / k

    def get_xcoords(self):
        return np.array(self.points)[:,0]

    def get_ycoords(self):
        return np.array(self.points)[:,1]

    @classmethod
    def from_great_circle_radius(cls, centroid, radius, num_points=10):
        """
        Generates a polygon object from a given radius and centroid location.

        Args:
            centroid: (lon, lat)
            radius: should be in (meters)
            num_points: more points is higher resolution polygon

        Returns:
            polygon
        """
        geod = pyproj.Geod(ellps='WGS84')
        azim = np.linspace(0, 360, num_points)
        # create vectors with same length as azim for computations
        center_lons = np.ones(num_points) * centroid[0]
        center_lats = np.ones(num_points) * centroid[1]
        radius = np.ones(num_points) * radius
        # get new lons and lats
        endlon, endlat, backaz = geod.fwd(center_lons, center_lats, azim, radius)
        # class method
        return cls(np.column_stack([endlon, endlat]))

class CartesianGrid2D:
    """Represents a 2D cartesian gridded region.

    The class provides functions to query onto an index 2D Cartesian grid and maintains a mapping between space coordinates defined
    by polygons and the index into the polygon array.

    Custom regions can be easily created by using the from_polygon classmethod. This function will accept an arbitrary closed
    polygon and return a CartesianGrid class with only points inside the polygon to be valid.
    """
    def __init__(self, polygons, dh, name='cartesian2d'):
        self.polygons = polygons
        self.dh = dh
        self.name = name
        a, xs, ys = _build_bitmask_vec(self.polygons, dh)
        # in mask, True = bad value and False = good value
        self.mask = a[:,:,0]
        # contains the mapping from polygon_index to the mask
        self.idx_map = a[:,:,1]
        # index values of polygons array into the 2d cartesian grid, based on the midpoint.
        self.xs = xs
        self.ys = ys

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
        if numpy.any(self.mask[idy,idx] == 1):
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
        mask = self.mask[idy, idx].astype(bool)
        # manually set values outside region
        mask[bad_idx] = True
        return mask

    def get_cartesian(self, data):
        """Returns 2d ndrray representation of the data set, corresponding to the bounding box.

        Args:
            data:
        """
        assert len(data) == len(self.polygons)
        results = numpy.zeros(self.mask.shape[:2])
        ny = len(self.ys)
        nx = len(self.xs)
        for i in range(ny):
            for j in range(nx):
                if self.mask[i, j] == 0:
                    idx = int(self.idx_map[i, j])
                    results[i, j] = data[idx]
                else:
                    results[i, j] = numpy.nan
        return results

    def get_bbox(self):
        """ Returns rectangular bounding box around region. """
        return (self.xs.min(), self.xs.max(), self.ys.min(), self.ys.max())

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
            'polygons': [{'lat': float(poly.origin[1]), 'lon': float(poly.origin[0])} for poly in self.polygons]
        }
        return adict

    @classmethod
    def from_dict(cls, adict):
        raise NotImplementedError("Todo!")

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

