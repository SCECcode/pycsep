import itertools
import os
from itertools import compress
from xml.etree import ElementTree as ET

import numpy

from csep.utils.basic_types import Polygon
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.core.spatial import compute_vertices, CartesianGrid2D, compute_vertex


def magnitude_bins(start_magnitude, end_magnitude, dmw):
    """ Wrapping function to return a numpy.ndarray of magnitude bin edges """
    return numpy.arange(start_magnitude, end_magnitude+dmw/2, dmw)

def create_space_magnitude_region(region, magnitudes):
    """ Very simple function to create space-magnitude region """
    if not isinstance(region, CartesianGrid2D):
        raise TypeError("region must be CartesianGrid2D")
    # bind to region class
    region.magnitudes = magnitudes
    region.num_mag_bins = len(region.magnitudes)
    return region

def california_relm_region(dh_scale=1):
    """
    Takes a CSEP1 XML file and returns a 'region' which is a list of polygons. This region can
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
    relm_region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh, name="california")
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