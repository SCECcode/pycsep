import itertools
import time
import os
import datetime

# third-party imports
import numpy

from csep.utils.log import LoggingMixin
from csep.core.regions import CartesianGrid2D, Polygon, create_space_magnitude_region
from csep.utils.calc import bin1d_vec
from csep.utils.time_utils import decimal_year, datetime_to_utc_epoch
from csep.core.catalogs import AbstractBaseCatalog
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR
from csep.utils.plots import plot_spatial_dataset


# idea: should this be a SpatialDataSet and the class below SpaceMagnitudeDataSet
# idea: this needs to handle non-carteisan regions, so maybe (lons, lats) should be a single variable like locations
# note: these are specified to 2D data sets and some minor refactoring needs to happen here.
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
        """ Returns a sum of the forecast data """
        return self.sum()

    def sum(self):
        """ Sums over all of the forecast data"""
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
        """ Returns the latitude of the lower left node of the spatial grid"""
        return self.region.origins()[:,1]

    def get_longitudes(self):
        """ Returns the lognitude of the lower left node of the spatial grid """
        return self.region.origins()[:,0]

    @property
    def polygons(self):
        return self.region.polygons

    def get_index_of(self, lons, lats):
        """ Returns the index of lons, lats in spatial region

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
        self.region = create_space_magnitude_region(self.region, magnitudes)

    @property
    def magnitudes(self):
        return self.region.magnitudes

    @property
    def min_magnitude(self):
        """ Returns the lowest magnitude bin edge """
        return numpy.min(self.magnitudes)

    @property
    def num_mag_bins(self):
        return len(self.magnitudes)

    def get_magnitudes(self):
        """ Returns the left edge of the magnitude bins. """
        return self.magnitudes

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
        """ Returns counts of events in magnitude bins """
        return numpy.sum(self.data, axis=0)

    def get_magnitude_index(self, mags):
        """ Returns the indices into the magnitude bins of selected magnitudes

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
    """ Class to represent grid-based forecasts """

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
        """ Creates MarkedGriddedDataSet class from custom parsing function.

        Args:
            func (callable): function will be called as func(*func_args).
            func_args (tuple): arguments to pass to func
            **kwargs: keyword arguments to pass to the GriddedForecast class constructor.

        Returns:
            :class:`csep.core.forecasts.GriddedForecast`: forecast object

        Note:
            The loader function `func` needs to return a tuple that contains (data, region, magnitudes). data is a
            :class:`numpy.ndarray`, region is a :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>`, and
            magnitudes are a :class:`numpy.ndarray` consisting of the magnitude bin edges. See the function
            :meth:`load_ascii<csep.core.forecasts.GriddedForecast.load_ascii>` for an example.

        """
        data, region, magnitudes = func(*func_args)
        # try to ensure that data are region are compatible with one another, but we can only rely on heuristics
        return cls(data=data, region=region, magnitudes=magnitudes, **kwargs)

    @classmethod
    def load_ascii(cls, ascii_fname, start_date=None, end_date=None, name=None):
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
        # this is very ugly, but since unique returns a sorted list, we want to get the index, sort that and then return
        # from the original array. same for magnitudes below.
        all_polys = data[:,:4]
        sorted_idx = numpy.sort(numpy.unique(all_polys, return_index=True, axis=0)[1], kind='stable')
        unique_poly = all_polys[sorted_idx]
        # create magnitudes bins using Mag_0, ignoring Mag_1 bc they are regular until last bin. we dont want binary search for this
        all_mws = data[:,-4]
        sorted_idx = numpy.sort(numpy.unique(all_mws, return_index=True)[1], kind='stable')
        mws = all_mws[sorted_idx]
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
        rates = data[:,-2].reshape(n_poly, n_mag_bins)
        # create / return class
        if name is None:
            name = os.path.basename(ascii_fname[:-4])
        gds = cls(start_date, end_date, magnitudes=mws, name=name, region=region, data=rates)
        return gds

    def plot(self, show=False, log=True, plot_args=None):
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
            time = 'forecast period'
        else:
            start = decimal_year(self.start_time)
            end = decimal_year(self.end_time)
            time = f'{round(end-start,3)} years'

        plot_args = plot_args or {}
        plot_args.setdefault('figsize', (10, 10))
        plot_args.setdefault('title', self.name)

        # this call requires internet connection and basemap
        if log:
            plot_args.setdefault('clabel', f'log10 M{self.min_magnitude}+ rate per {str(dh)}° x {str(dh)}° per {time}')
            with numpy.errstate(divide='ignore'):
                ax = plot_spatial_dataset(numpy.log10(self.spatial_counts(cartesian=True)), self.region, show=show, plot_args=plot_args)
        else:
            plot_args.setdefault('clabel', f'M{self.min_magnitude}+ rate per {str(dh)}° x {str(dh)}° per {time}')
            ax = plot_spatial_dataset(self.spatial_counts(cartesian=True), self.region, show=show, plot_args=plot_args)
        return ax


class CatalogForecast(LoggingMixin):
    """ Catalog based forecast defined as a family of stochastic event sets. """

    def __init__(self, filename=None, catalogs=None, name=None,
                 filter_spatial=False, filters=None, apply_mct=False,
                 region=None, expected_rates=None, start_time=None, end_time=None,
                 n_cat=None, event=None, loader=None, catalog_type='ascii',
                 catalog_format='native'):


        """
        The region information can be provided along side the data, if they are stored in one of the supported file formats.
        It is assumed that the region for each data is identical. If the regions are not provided with the data files,
        they must be provided explicitly. The california testing region can be loaded using :func:`csep.utils.spatial.california_relm_region`.

        There are a few different ways this class can be constructed, each

        The region is not required to load a forecast or to perform basic operations on a forecast, such as counting events.
        Any binning of events in space or magnitude will require a spatial region or magnitude bin definitions, respectively.

        Args:
            filename (str): Path to the file or directory containing the forecast.
            catalogs: iterable of :class:`csep.core.catalogs.AbstractBaseCatalog`
            filter_spatial (bool): if true, will filter to area defined in space region
            apply_mct (bool): this should be provided if a time-dependent magnitude completeness model should be
                              applied to the forecast
            filters (iterable): list of data filter strings. these override the filter_magnitude and filter_time arguments
            region: :class:`csep.core.spatial.CartesianGrid2D` including magnitude bins
            start_time (datetime.datetime): start time of the forecast
            end_time (datetime.datetime): end time of the forecast
            name (str): name of the forecast, will be used for defaults in plotting and other places
            n_cat (int): number of catalogs in the forecast
            event (:class:`csep.models.Event`): if the forecast is associated with a particular event

        """

        super().__init__()

        # used for labeling plots, filenames, etc, should be human readable
        self.name = name

        # path to forecast location
        self.filename = filename

        # should be iterable
        self.catalogs = catalogs or []

        # should be a generator function
        self.loader = loader

        # used if the forecast is associated with a particular event
        self.event = event

        # these can be used to filter catalogs to a desired experiment region
        self.filters = filters or []

        self.filter_spatial = filter_spatial
        self.apply_mct = apply_mct

        # data format used for loading catalogs
        self.catalog_type = catalog_type
        self.catalog_format = catalog_format

        # should be a MarkedGriddedDataSet
        self.expected_rates = expected_rates

        # defines the space, time, and magnitude region of the forecasts
        self.region = region

        # start and end time of the forecast
        self.start_time = start_time
        self.end_time = end_time

        # time horizon in years
        if self.start_time is not None and self.end_time is not None:
            self.time_horizon_years = (self.end_epoch - self.start_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000
            # add time filters only if filters are not provied and user wants to filter in time

        # number of simulated catalogs
        self.n_cat = n_cat

        # used to handle the iteration over catalogs
        self._idx = 0

        # load catalogs if catalogs aren't provided, this might be a generator
        if not self.catalogs:
            self._load_catalogs()

    def __iter__(self):
        return self

    def __next__(self):
        """ Allows the class to be used in a for-loop. Handles the case where the catalogs are stored as a list or
        loaded in using a generator function. The latter solves the problem where memory is a concern or all of the
        catalogs should not be held in memory at once. """
        try:
            n_items = len(self.catalogs)
            assert self.n_cat == n_items
            # here, we have reached the end of the list, simply reset the index to the front
            if self._idx >= self.n_cat:
                self._idx = 0
                raise StopIteration()
            catalog = self.catalogs[self._idx]
            self._idx += 1
        except TypeError:
            # handle generator case. a generator does not have the __len__ attribute, but an iterable does.
            try:
                catalog = next(self.catalogs)
                self._idx += 1
            except StopIteration:
                # gets a new generator function after the old one is exhausted
                self.catalogs = self.loader(format=self.catalog_format, filename=self.filename,
                                            region=self.region, name=self.name)
                self.n_cat = self._idx
                self._idx = 0
                raise StopIteration()
        # apply filtering to catalogs, these can throw errors if not configured properly
        if self.filters:
            catalog = catalog.filter(self.filters)
        if self.apply_mct:
            catalog = catalog.apply_mct(self.event.magnitude, datetime_to_utc_epoch(self.event.time))
        if self.filter_spatial:
            catalog = catalog.filter_spatial(self.region)
        # return potentially filtered data
        return catalog

    def _load_catalogs(self):
        self.catalogs = self.loader(format=self.catalog_format, filename=self.filename, region=self.region, name=self.name)

    @property
    def start_epoch(self):
        return datetime_to_utc_epoch(self.start_time)

    @property
    def end_epoch(self):
        return datetime_to_utc_epoch(self.end_time)

    @property
    def magnitudes(self):
        """ Returns left bin-edges of magnitude bins """
        return self.region.magnitudes

    @property
    def min_magnitude(self):
        """ Returns smallest magnitude bin edge of forecast """
        return numpy.min(self.region.magnitudes)

    def spatial_counts(self, cartesian=False):
        """ Returns the expected spatial counts from forecast """
        if self.expected_rates is not None:
            return self.expected_rates.spatial_counts(cartesian=cartesian)
        else:
            return None

    def magnitude_counts(self):
        """ Returns expected magnitude counts from forecast """
        if self.expected_rates is not None:
            return self.expected_rates.magnitude_counts()
        else:
            return None

    def get_expected_rates(self, verbose=False, return_skipped=False):
        """ Compute the expected rates in space-magnitude bins

        Args:
            catalogs_iterable (iterable): collection of catalogs, should be filtered outside the function
            data (csep.core.AbstractBaseCatalog): observation data

        Return:
            :class:`csep.core.forecasts.GriddedForecast`
            list of tuple(lon, lat, magnitude) events that were skipped in binning. if data was filtered in space
            and magnitude beforehand this list shoudl be empty.
        """
        # self.n_cat might be none here, if catalogs haven't been loaded and its not yet specified.
        skipped_list = []
        if self.region is None or self.region.magnitudes is None:
            raise AttributeError("Forecast must have space-magnitude regions to compute expected rates.")
        # need to compute expected rates, else return.
        if self.expected_rates is None:
            t0 = time.time()
            data = numpy.empty([])
            for i, cat in enumerate(self):
                # compute spatial density from each data, force data region to use the forecast region
                cat.region = self.region
                gridded_counts, skipped = cat.spatial_magnitude_counts(ret_skipped=True)
                skipped_list.extend(skipped)
                if i == 0:
                    data = numpy.array(gridded_counts)
                else:
                    data += numpy.array(gridded_counts)
                # output status
                if verbose:
                    tens_exp = numpy.floor(numpy.log10(i + 1))
                    if (i + 1) % 10 ** tens_exp == 0:
                        t1 = time.time()
                        print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)
            # after we iterate through the catalogs, we know self.n_cat
            data = data / self.n_cat
            self.expected_rates = GriddedForecast(self.start_time, self.end_time, data=data, region=self.region,
                                                  magnitudes=self.magnitudes, name=self.name)
            if return_skipped:
                return (self.expected_rates, skipped_list)
            else:
                return self.expected_rates

    def get_dataframe(self):
        """Return a single dataframe with all of the events from all of the catalogs."""
        raise NotImplementedError("get_dataframe is not implemented.")

    def write_ascii(self, fname, header=True, loader=None ):
        """ Writes data forecast to ASCII format


        Args:
            fname (str): Output filename of forecast
            header (bool): If true, write header information; else, do not write header.


        Returns:
            NoneType
        """

        raise NotImplementedError('write_ascii is not implemented!')

    @classmethod
    def load_ascii(cls, fname, **kwargs):
        """ Loads ASCII format for data forecast.

        Args:
            fname (str): path to file or directory containing forecast files

        Returns:
              :class:`csep.core.forecasts.CatalogForecast
        """
        raise NotImplementedError("load_ascii is not implemented!")