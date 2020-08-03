import itertools
import os
import numpy
import datetime
import decimal
from csep.utils.log import LoggingMixin
from csep.utils.spatial import CartesianGrid2D
from csep.utils.basic_types import Polygon
from csep.utils.calc import bin1d_vec
from csep.utils.time_utils import decimal_year
from csep.core.catalogs import AbstractBaseCatalog
from csep.utils.constants import CSEP_MW_BINS
from csep.utils.plots import plot_spatial_dataset

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

        # note: do not access this member through _data, always use .data.
        self._data = data
        self.region = region
        self.name = name

        # this value lets us scale the forecast without much additional memory constraints and makes the calls
        # idempotent
        self._scale = 1

    @property
    def data(self):
        """ Contains the spatio-magnitude forecast as 2d numpy.ndarray.

        The dimensions of this array are (num_spatial_bins, num_magnitude_bins). The spatial bins can be indexed through
        a look up table as part of the region class. The magnitude bins used are stored as directly as an attribute of
        class.
        """
        return self._data * self._scale

    @property
    def event_count(self):
        return numpy.sum(self.data)

    def spatial_counts(self, cartesian=False):
        """ Returns the counts (or rates) of earthquakes within each spatial bin.

        Args:
            cartesian (bool): if true, will return a 2d grid representing the bounding box of the forecast

        Returns:
            ndarray containing the count in each bin

        """
        if cartesian:
            return self.region.get_cartesian(self.data)
        else:
            return self.data

    def get_latitudes(self):
        return self.region.origins()[:,1]

    def get_longitudes(self):
        return self.region.origins()[:,0]

    @property
    def polygons(self):
        return self.region.polygons

    def get_index_of(self, lons, lats):
        """ Returns the index of lons, lats in self.region.polygons.

        See csep.utils.spatial.CartesianGrid2D for more details.

        Args:
            lons: ndarray-like
            lats: ndarray-like

        Returns:
            idx: ndarray-like

        Raises:
            ValueError: if lons or lats are outside of the region.
        """
        return self.region.get_index_of(lons, lats)

    def scale(self, val):
        """Scales forecast by floating point value.

        Args:
            val: int, float, or ndarray. This value multiplicatively scale the values in forecast. Use a value of
                 1 to recover original value of the forecast.

        Raises:
            ValueError: must be int, float, or ndarray
        """
        self._scale = val
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

    def __init__(self, magnitudes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnitudes = magnitudes

    @property
    def min_magnitude(self):
        return numpy.min(self.magnitudes)

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

    def get_magnitude_index(self, mags):
        """ Returns the indices into the magnitude bins corresponding to the values in mags.

        Note: the right-most bin is treated as extending to infinity.

        Args:
            mags (array-like): list of magnitudes

        Returns:
            idm (array-like): indices corresponding to mags

        Raises:
            ValueError
        """
        idm = bin1d_vec(mags, self.magnitudes, right_continuous=True)
        if numpy.any(idm == -1):
            raise ValueError("mags outside the range of forecast magnitudes.")
        return idm

class GriddedForecast(MarkedGriddedDataSet):

    def __init__(self, start_time=None, end_time=None, *args, **kwargs):
        """
        Constructor for GriddedForecast class

        Args:
            start_time (datetime.datetime):
            end_time (datetime.datetime):
        """
        super().__init__(*args, **kwargs)
        self.start_time = start_time
        self.end_time = end_time

    def scale_to_test_date(self, test_datetime):
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
        res = self.scale(fore_frac)
        return res

    def target_event_rates(self, target_catalog, scale=True):
        """ Generates data set of target event rates given a target data.

        The data should already be scaled to the same length as the forecast time horizon. Explicit checks for these
        cases are not conducted in this function.

        If scale=True then the target event rates will be scaled down to the rates for one day. This choice of time
        can be made without a loss of generality. Please see Rhoades, D. A., D. Schorlemmer, M. C. Gerstenberger,
        A. Christophersen, J. D. Zechar, and M. Imoto (2011). Efficient testing of earthquake forecasting models,
        Acta Geophys 59 728-747.

        Args:
            target_catalog (csep.core.data.AbstractBaseCatalog): data containing target events
            scale (bool): if true, rates will be scaled to one day.

        Returns:
            out (tuple): target_event_rates, n_fore. target event rates are the

        """
        if not isinstance(target_catalog, AbstractBaseCatalog):
            raise TypeError("target_catalog must be csep.core.data.AbstractBaseCatalog class.")

        if scale:
            # first get copy so we dont contaminate the rates of the forecast, this can be quite large for global files
            # if we run into memory problems, we can implement a sparse form of the forecast.
            data = numpy.copy(self.data)
            # straight-forward implementation, relies on correct start and end time
            elapsed_days = (self.end_time - self.start_time).days
            # scale the data down to days
            data = data / elapsed_days
        else:
            # just pull reference to stored data
            data = self.data

        # get longitudes and latitudes of target events
        lons = target_catalog.get_longitudes()
        lats = target_catalog.get_latitudes()
        mags = target_catalog.get_magnitudes()

        # this array does not keep track of any location anymore. however, it can be computed using the data again.
        rates = self.get_rates(lons, lats, mags, data=data)
        # we return the sum of data, bc data might be scaled within this function
        return rates, numpy.sum(data)

    def get_rates(self, lons, lats, mags, data=None, ret_inds=False):
        """ Returns the rate associated with a longitude, latitude, and magnitude.

        Args:
            lon: longitude of interest
            lat: latitude of interest
            mag: magnitude of interest
            data: optional, if not none then use this data value provided with the forecast

        Returns:
            rates (float or ndarray)

        Raises:
            RuntimeError: lons lats and mags must be the same length
        """
        if len(lons) != len(lats) and len(lats) != len(mags):
            raise RuntimeError("lons, lats, and mags must have the same length.")
        # get index of lon and lat value, if lats, lons, not in region raise value error
        idx = self.get_index_of(lons, lats)
        # get index of magnitude bins, if lats, lons, not in region raise value error
        idm = self.get_magnitude_index(mags)
        # retrieve rates from internal data structure
        if data is None:
            rates = self.data[idx,idm]
        else:
            rates = data[idx,idm]
        if ret_inds:
            return rates, (idx, idm)
        else:
            return rates

    @classmethod
    def from_custom(cls, func, func_args=(), **kwargs):
        """Creates MarkedGriddedDataSet class from custom parsing function.

        Custom parsing function should return a tuple containing the forecast data as appropriate numpy.ndarray and
        region class. We can only rely on some heuristics to ensure that these classes are set up appropriately.
        """
        data, region, magnitudes = func(*func_args)
        # try to ensure that data are region are compatible with one another, but we can only rely on heuristics
        return cls(data=data, region=region, magnitudes=magnitudes, **kwargs)

    @classmethod
    def from_csep1_ascii(cls, ascii_fname, start_date=None, end_date=None):
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
        dh = float(unique_poly[0,3] - unique_poly[0,2])
        # create CarteisanGrid of points
        region = CartesianGrid2D([Polygon(bbox) for bbox in bboxes], dh)
        # get dims of 2d np.array
        n_mag_bins = len(mws)
        n_poly = region.num_nodes
        # reshape rates into correct 2d format
        rates = data[:,-2].reshape(n_poly,n_mag_bins)
        # create / return class
        gds = cls(start_date, end_date, magnitudes=mws, name=os.path.basename(ascii_fname[:-4]), region=region, data=rates)
        return gds

    def plot(self, show=False, plot_args=None):
        """ Plot gridded forecast according to plate-carree proejction

        Args:
            show (bool): if true, show the figure. this call is blocking.
            plot_args (optional/dict): dictionary containing plotting arguments for making figures

        Returns:
            axes: matplotlib.Axes.axes
        """
        # no mutable function arguments
        dh = round(self.region.dh, 5)
        if self.start_time is None or self.end_time is None:
            time = 'year'
        else:
            start = decimal_year(self.start_time)
            end = decimal_year(self.end_time)
            time = f'{start-end} years'

        plot_args = plot_args or {}
        plot_args.setdefault('figsize', (9, 9))
        plot_args.setdefault('title', self.name)
        plot_args.setdefault('clabel', f'Eq. rate per {str(dh)}° x {str(dh)}° per {time}')
        # this call requires internet connection and basemap
        ax = plot_spatial_dataset(self.spatial_counts(cartesian=True), self.region, show=show, plot_args=plot_args)
        return ax

    def number_test(self, catalog):
        pass

    def magnitude_test(self, catalog):
        pass

    def conditional_likelihood_test(self, catalog):
        pass



class CatalogForecast(LoggingMixin):

    def __init__(self, catalogs=None, region=None, magnitudes=CSEP_MW_BINS):
        super().__init__()
        self.catalogs = catalogs or []
        self.region = region
        self.magnitudes = magnitudes
        self.origin_epoch = None
        self.end_epoch = None

    @property
    def num_mag_bins(self):
        return

    def spatial_counts(self):
        pass

    def magnitude_counts(self):
        pass

    def spatial_magnitude_counts(self):
        pass

