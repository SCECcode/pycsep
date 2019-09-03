import numpy as np
import matplotlib

import redis
import pickle
import pyproj

from queue import Empty, Full
from multiprocessing import Queue, Array

from csep.utils.calc import discretize

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


class ArrayView:
    def __init__(self, array, max_bytes, dtype, el_shape, i_item=0):
        self.dtype = dtype
        self.el_shape = el_shape
        self.nbytes_el = self.dtype.itemsize * np.product(self.el_shape)
        self.n_items = int(np.floor(max_bytes / self.nbytes_el))
        self.total_shape = (self.n_items,) + self.el_shape
        self.i_item = i_item
        self.view = np.frombuffer(array, dtype, np.product(self.total_shape)).reshape(self.total_shape)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.el_shape == other.el_shape and self.dtype == other.dtype
        return False

    def push(self, element):
        self.view[self.i_item, ...] = element
        i_inserted = self.i_item
        self.i_item = (self.i_item + 1) % self.n_items
        return self.dtype, self.el_shape, i_inserted # a tuple is returned to maximise performance

    def pop(self, i_item):
        return self.view[i_item, ...]

    def fits(self, item):
        if isinstance(item, np.ndarray):
            return item.dtype == self.dtype and item.shape == self.el_shape
        return (item[0] == self.dtype and
                item[1] == self.el_shape and
                item[2] < self.n_items)


class ArrayQueue:
    """ A drop-in replacement for the multiprocessing queue, usable
     only for numpy arrays, which removes the need for pickling and
     should provide higher speeds and lower memory usage
    """
    def __init__(self, max_mbytes=10):
        self.maxbytes = int(max_mbytes*1000000)
        self.array = Array('c', self.maxbytes)
        self.view = None
        self.queue = Queue()
        self.read_queue = Queue()
        self.last_item = 0

    def check_full(self):
        while True:
            try:
                self.last_item = self.read_queue.get(timeout=0.00001)
            except Empty:
                break
        if self.view.i_item == self.last_item:
            raise Full("Queue of length {} full when trying to insert {},"
                       " last item read was {}".format(self.view.n_items,
                                                       self.view.i_item, self.last_item))

    def put(self, element):
        if self.view is None or not self.view.fits(element):
            self.view = ArrayView(self.array.get_obj(), self.maxbytes,
                                  element.dtype, element.shape)
            self.last_item = 0
        else:
            self.check_full()
        qitem = self.view.push(element)

        self.queue.put(qitem)

    def get(self, **kwargs):
        aritem = self.queue.get(**kwargs)
        if self.view is None or not self.view.fits(aritem):
            self.view = ArrayView(self.array.get_obj(), self.maxbytes,
                                  *aritem)
        self.read_queue.put(aritem[2])
        return self.view.pop(aritem[2])

    def clear(self):
        """ Empties the queue without the need to read all the existing
        elements
        :return: nothing
        """
        self.view = None
        while True:
            try:
                it = self.queue.get_nowait()
            except Empty:
                break
        while True:
            try:
                it = self.read_queue.get_nowait()
            except Empty:
                break

        self.last_item = 0

    def empty(self):
        return self.queue.empty()

class RedisQueue:
    """Simple Queue with Redis Backend"""

    def __init__(self, name, serializer=pickle, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db = redis.Redis(**redis_kwargs)
        self.serializer = serializer
        self.key = '%s:%s' % (namespace, name)

    def __len__(self):
        return self.__db.llen(self.key)


    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)


    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def clear(self):
        """ Clear the queue of all messages """
        self.__db.delete(self.key)

    def consume(self, **kwargs):
        kwargs.setdefault('block', True)
        try:
            while True:
                msg = self.get(**kwargs)
                if msg is None:
                    break
        except KeyboardInterrupt:
            print; return
        yield msg


    def put(self, item, protocol=4):
        """Put item into the queue."""
        if self.serializer is not None:
            msg = self.serializer.dumps(item, protocol=protocol)
        self.__db.rpush(self.key, msg)


    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue."""
        if block:
            if timeout is None:
                timeout = 0
            msg = self.__db.blpop(self.key, timeout=timeout)
            if msg is not None:
                msg = msg[1]
        else:
            msg = self.__db.lpop(self.key)
        if msg is not None and self.serializer is not None:
            msg = self.serializer.loads(msg)
        return msg


    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)