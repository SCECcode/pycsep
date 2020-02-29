import itertools

import numpy
import copy
import datetime
from csep.utils.log import LoggingMixin
from csep.utils.spatial import CartesianGrid2D
from csep.utils.basic_types import Polygon
from csep.utils.time_utils import decimal_year

class GriddedDataSet(LoggingMixin):
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

    def __init__(self, data=None, region=None, name=None):
        """ Constructs GriddedSeismicity class.

        Attributes:
            data (numpy.ndarray): numpy.ndarray
            region:
            name:
            time_horizon:
        """
        super().__init__()
        self._data = data
        self.region = region
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

    @property
    def polygons(self):
        return self.region.polygons


    def scale(self, val, in_place=False):
        """Scales forecast by floating point value.

        Args:
            val: int, float, or ndarray. This value multiplicatively scale the values in forecast.

        Raises:
            ValueError: must be int, float, or ndarray
        """
        if not isinstance(val, (int, float, numpy.ndarray)):
            raise ValueError("scaling value must be (int, float, or numpy.ndarray).")
        if not in_place:
            new = copy.deepcopy(self)
            new._data *= val
            return new
        else:
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

    def __init__(self, magnitudes=None, time_horizon=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_horizon = time_horizon
        self.magnitudes = magnitudes

    @property
    def num_mag_bins(self):
        return len(self.magnitudes)

    @property
    def num_nodes(self):
        return self.region.num_nodes

    def spatial_counts(self, cartesian=False):
        """
        Integrates over magnitudes to return the spatial version of the forecast.

        Args:
            cartesian (bool): if true, will return a 2d grid representing the bounding box of the forecast

        Returns:
            ndarray containing the count in each bin

        """
        if cartesian:
            return self.region.get_cartesian(numpy.sum(self.data, axis=1))
        else:
            return numpy.sum(self.data, axis=1)

    def magnitude_counts(self):
        return numpy.sum(self.data, axis=0)


class GriddedForecast(MarkedGriddedDataSet):

    def __init__(self, start_time, end_time, *args, **kwargs):
        """
        Constructor for GriddedForecast class

        Args:
            start_time (datetime.datetime):
            end_time (datetime.datetime):
        """
        super().__init__(*args, **kwargs)
        self.start_time = start_time
        self.end_time = end_time

    def scale_to_test_date(self, test_datetime, in_place=True):
        """ Scales forecast data by the fraction of the date.

        Uses the concept of decimal years to keep track of leap years. See the csep.utils.time_utils.decimal_year for
        details on the implementation. If datetime is before the start_date or after the end_date, we will scale the
        forecast by unity.

        These datetime objects can be timezone aware in UTC timezone or both not time aware. This function will raise a
        TypeError according to the specifications of datetime module if these conditions are not met.

        Args:
            test_datetime (datetime.datetime): date to scale the forecast
            in_place (bool): if false, creates a deep copy of the object and scales that instead
        """

        # Note: this will throw a TypeError if datetimes are not either both time aware or not time aware.
        if test_datetime >= self.end_time:
            return self

        if test_datetime <= self.start_time:
            return self

        fore_dur = decimal_year(self.end_time) - decimal_year(self.start_time)
        # we are adding one day, bc tests are considered to occur at the end of the day specified by test_datetime.
        test_date_dec = decimal_year(test_datetime + datetime.timedelta(1))
        fore_frac = (test_date_dec - decimal_year(self.start_time)) / fore_dur
        res = self.scale(fore_frac, in_place=in_place)
        return res

    @classmethod
    def from_custom(cls, afunc, start_date, end_date, name=None, time_horizon=None, *args, **kwargs):
        """Creates MarkedGriddedDataSet class from custom parsing function.

        Custom parsing function should return a tuple containing the forecast data as appropriate numpy.ndarray and
        region class. We can only rely on some heuristics to ensure that these classes are set up appropriately.  """
        data, region, magnitudes = afunc(*args, **kwargs)
        # try to ensure that data are region are compatible with one another, but we can only rely on heuristics
        return cls(start_date, end_date, data=data, region=region, magnitudes=magnitudes, name=name, time_horizon=time_horizon)

    @classmethod
    def from_csep1_ascii(cls, ascii_fname, start_date, end_date):
        """ Reads Forecast file from CSEP1 ascii format.

        The ascii format from CSEP1 testing centers. The ASCII format does not contain headers. The format is listed here:

        Lon_0, Lon_1, Lat_0, Lat_1, z_0, z_1, Mag_0, Mag_1, Rate, Flag

        For the purposes of defining region objects and magnitude bins use the Lat_0 and Lon_0 values along with Mag_0.
        We can assume that the magnitude bins are regularly spaced to allow us to compute Deltas.

        The file is row-ordered so that magnitude bins are fastest then followed by space.

        Args:
            ascii_fname: file name of csep forecast in .dat format
        """
        # Load data
        data = numpy.loadtxt(ascii_fname)
        unique_poly = numpy.unique(data[:,:4], axis=0)
        # create magnitudes bins using Mag_0, ignoring Mag_1 bc they are regular until last bin. we dont want binary search for this
        mws = numpy.unique(data[:,-4])
        # csep1 stores the lat lons as min values and not (x,y) tuples
        bboxes = [tuple(itertools.product(bbox[:2], bbox[2:])) for bbox in unique_poly]
        # the spatial cells are arranged fast in latitude, so this only works for the specific csep1 file format
        dh = unique_poly[0,3] - unique_poly[0,2]
        # create CarteisanGrid of points
        region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh)
        # get dims of 2d np.array
        n_mag_bins = len(mws)
        n_poly = region.num_nodes
        # reshape rates into correct 2d format
        rates = data[:,-2].reshape(n_poly,n_mag_bins)
        # create / return class
        gds = cls(start_date, end_date, magnitudes=mws, name=ascii_fname[:-4], region=region, data=rates)
        return gds