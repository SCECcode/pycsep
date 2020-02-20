import numpy

from csep.utils.calc import bin1d_vec


class GriddedDataSet:
    """Represents space-magnitude discretized seismicity implementation.

    Map-based and discrete forecasts, such as those provided by time-independent forecasts can be stored using this format.
    This object will provide some convenience routines and functionality around the numpy.ndarray primative that
    actually stores the space-time magnitude forecast.

    Earthquake forecasts based on discrete space-magnitude regions are read into this format by default. This format can
    support multiple types of region including 2d and 3d cartesian meshes. The appropriate region must be provided. By default
    the magniutde is always treated as the 'fast' dimension of the numpy.array.

    Attributes:
        data (numpy.ndarray): 2d numpy.ndarray containing the spatial and magnitude bins with magnitudes being the fast dimension.
        region: csep.utils.spatial.2DCartesianGrid class containing the mapping of the data points into the region.
        mags: list or numpy.ndarray class containing the lower (inclusive) magnitude values from the discretized
              magnitudes. The magnitude bins should be regularly spaced.
    """

    def __init__(self, data=None, region=None, name=None, time_horizon=None):
        """ Constructs GriddedSeismicity class.

        Attributes:
            data (numpy.ndarray): numpy.ndarray
            grids (tuple): tuple of
        """
        self._data = data
        # metadata associated with spatial region
        self.region = region
        # discretization of the magnitude bins
        self.time_horizon = time_horizon
        self.name = name

    @property
    def data(self):
        """ Contains the spatio-magnitude forecast as 2d numpy.ndarray.

        The dimensions of this array are (num_spatial_bins, num_magnitude_bins). The spatial bins can be indexed through
        a look up table as part of the region class. The magnitude bins used are stored as directly as an attribute of
        class.
        """
        return self._data

    @property
    def event_count(self):
        return numpy.sum(self.data)

    @property
    def latitudes(self):
        return self.region.origins()[:,1]

    @property
    def longitudes(self):
        return self.region.origins()[:,0]

    def scale(self, val):
        """Scales forecast by floating point value.

        Args:
            val: int, float, or ndarray. This value multiplicatively scale the values in forecast.

        Raises:
            ValueError: must be int, float, or ndarray
        """
        if not isinstance(val, (int, float, numpy.ndarray)):
            raise ValueError("scaling value must be (int, float, or numpy.ndarray).")
        self._data *= val
        return self

    def to_dict(self):
        return

    @classmethod
    def from_dict(cls, adict):
        raise NotImplementedError()


class MarkedGriddedDataSet(GriddedDataSet):
    """
    Represents a gridded forecast in CSEP. The data must be stored as a 2D numpy array where the fast column is magnitude.
    The shape of this array will have (n_space_bins, n_mag_bins) and can be large.

    """

    def __init__(self, magnitudes=None, name=None, time_horizon=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.time_horizon = time_horizon
        self.magnitudes = magnitudes

    @property
    def spatial_counts(self, cartesian=False):
        """
        Integrates over magnitudes to return the spatial version of the forecast.

        Args:
            cartesian (bool): if true, will return a 2d grid representing the bounding box of the forecast

        Returns:
            ndarray containing the count in each bin

        """
        if cartesian:
            return self.region.get_cartesian(self.spatial_counts, cartesian=False)
        else:
            return numpy.sum(self.data, axis=1)

    @property
    def magnitude_counts(self):
        return numpy.sum(self.data, axis=0)

    @classmethod
    def from_custom(cls, afunc, name=None, time_horizon=None, *args, **kwargs):
        """Creates MarkedGriddedDataSet class from custom parsing function.

        Custom parsing function should return a tuple containing the forecast data as appropriate numpy.ndarray and
        region class. We can only rely on some heuristics to ensure that these classes are set up appropriately.  """
        data, region, magnitudes = afunc(*args, **kwargs)
        # try to ensure that data are region are compatible with one another, but we can only rely on heuristics
        return cls(data=data, region=region, magnitudes=magnitudes, name=name, time_horizon=time_horizon)

    @classmethod
    def from_csep1_xml(cls, xml_fname):
        raise NotImplementedError()



