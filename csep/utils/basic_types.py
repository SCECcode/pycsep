import numpy as np
import matplotlib

import pyproj

from csep.utils.calc import discretize


def seq_iter(iterable):
    """
    helper function to handle iterating over a dict or list

    should iterate using:

    for idx in iterable:
         value = iterable[idx]
         ...

    Args:
        iterable: an iterable object

    Returns:
        key to access iterable

    """
    return iterable if isinstance(iterable, dict) else range(len(iterable))

class GriddedDataSet:
    """
    Allows us to work with data that need to be discretized and aggregated even though the the global min/max values
    are not known before hand.

    Using this function incurs some addition overhead, instead of simply just binning and combining.

    """
    def __init__(self, dh=0.1):
        self.dh = dh
        self.data = np.empty([])
        self.bins = []

    def add(self, data):
        # convert to integer to compute modulus
        int_data = int(np.min(data) * 1e12)
        int_dh = int(self.dh * 1e12)
        lower = (int_data - (int_data % int_dh)) / 1e12
        upper = lower + self.dh

        bins = np.arange(lower, upper+self.dh, self.dh)
        binned_data = discretize(data, bins)

        # bins fits nicely within self.bins
        self._merge(bins, binned_data)

    def _merge(self, bins, binned_data):

        if not self.bins:
            self.bins = bins
            self.data = binned_data

class Polygon:
    """
    Represents polygons defined through a collection of vertices.

    This polygon is assumed to be 2d, but could contain an arbitrary number of vertices.
    """
    def __init__(self, points):
        # instance members
        self.points = points
        self.origin = self.points[0]
        self.path = matplotlib.path.Path(self.points)

    def __str__(self):
        return str(self.origin)

    def contains(self, points):
        return self.path.contains_points(points)

    def centroid(self):
        """ return the cetroid of the polygon."""
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
        return cls(np.column_stack([endlon,endlat]))


