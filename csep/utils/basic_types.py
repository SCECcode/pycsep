import collections

import numpy as np
import matplotlib
import matplotlib.path

import pyproj

from csep.utils.calc import bin1d_vec


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


class AdaptiveHistogram:
    """
    Allows us to work with data that need to be discretized and aggregated even though the the global min/max values
    are not known before hand. Data are discretized according to the dh and anchor positions and their extreme values.
    If necessary the range of the bin_edges are expanded to accommodate new data

    Using this function incurs some addition overhead, instead of simply just binning and combining.

    """
    def __init__(self, dh=0.1, anchor=0.0):
        self.dh = dh
        self.anchor = anchor
        self.data = np.array([])
        self.bins = np.array([])

    def add(self, data):

        if len(data) == 0:
            return

        # float point arithmitic can be an issue here
        data_min = np.min(data)
        data_max = np.max(data)

        # need to know the range of the data to be inserted on discretized grid (min, max)
        # this is to determine the discretization of the data
        eps=np.finfo(np.float).eps
        disc_min = np.floor((data_min+eps-self.anchor)*self.rec_dh)/self.rec_dh+self.anchor
        disc_max = np.ceil((data_max+eps-self.anchor)*self.rec_dh)/self.rec_dh+self.anchor

        # compute new bin edges from data
        new_bins = np.arange(disc_min, disc_max+self.dh/2, self.dh)

        # merge data
        self._merge(new_bins, data)

    def _merge(self, bins, data):

        # 1) current bins dont exist
        if self.bins.size == 0:
            self.bins = bins
            self.data = np.zeros(len(self.bins))
            idx = bin1d_vec(data, self.bins)
            np.add.at(self.data, idx, 1)
            return

        # 2) new bins subset of current bins
        if bins[0] >= self.bins[0] and bins[-1] <= self.bins[-1]:
            idx = bin1d_vec(data, self.bins)
            np.add.at(self.data,idx,1)
            return

        # 3) new bins are outside current bins
        if bins[0] < self.bins[0]:
            bin_min = bins[0]
        else:
            bin_min = self.bins[0]

        if bins[-1] > self.bins[-1]:
            bin_max = bins[-1]
        else:
            bin_max = self.bins[-1]

        # generate new bins
        new_bins = np.arange(bin_min, bin_max+self.dh/2, self.dh)
        tmp_data = np.zeros(len(new_bins))
        # merge new data to new bins
        # get old bin locations relative to new bins
        idx = bin1d_vec(self.bins, new_bins)
        # add old data
        tmp_data[idx] = self.data
        self.data = tmp_data
        idx = bin1d_vec(data, new_bins)
        np.add.at(self.data, idx, 1)
        self.bins = new_bins
        return

    @property
    def rec_dh(self):
        return 1.0 / self.dh

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
        return cls(np.column_stack([endlon,endlat]))

def transpose_dict(adict):
    """Transposes a dict of dicts to regroup the data."""
    out = collections.defaultdict(dict)
    for k,v in adict.items():
        for ik,iv in v.items():
            out[ik][k] = iv
    return out
