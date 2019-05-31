import numpy
import scipy
import pandas
import datetime
import operator

# CSEP Imports
from csep.utils.time import epoch_time_to_utc_datetime, timedelta_from_years, datetime_to_utc_epoch
from csep.utils.comcat import search
from csep.utils.stats import min_or_none, max_or_none


class BaseCatalog:
    """
    Base class for CSEP2 catalogs.

    Todo:
        Come up with idea on how to manage the region of a catalog.
        Would be used for filtering or binning. shapely, geopandas for spatial and DataFrame for temporal.
    """
    def __init__(self, filename=None, catalog=None, catalog_id=None, format=None, name=None, region=None, compute_stats=False,
                 min_magnitude=None, max_magnitude=None,
                 min_latitude=None, max_latitude=None,
                 min_longitude=None, max_longitude=None,
                 start_time=None, end_time=None):

        self.filename = filename
        self.catalog_id = catalog_id
        self.format = format
        self.name = name
        self.region = region
        self.compute_stats = compute_stats

        # cleans the catalog to set as ndarray, see setter.
        self.catalog = catalog

        # class attributes that are not settable from constructor (adding here for readability)
        self.mfd = None

        # set these values from initially
        self.event_count = self.get_number_of_events()
        self.max_magnitude = max_magnitude
        self.min_magnitude = min_magnitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.start_time = start_time
        self.end_time = end_time

        # use user defined stats if entered into catalog
        try:
            if catalog is not None and self.compute_stats:
                self._update_catalog_stats()
        except (AttributeError, NotImplementedError):
            print('Warning: could not parse catalog statistics by reading catalog! get_magnitudes(), get_latitudes() and get_longitudes() ' +
                  'must be implemented and bound to calling class! Reverting to old values.')

    def __str__(self):
        s='''
        Name: {}

        Start Date: {}
        End Date: {}

        Latitude: ({:.2f}, {:.2f})
        Longitude: ({:.2f}, {:.2f})

        Min Mw: {:.2f}
        Max Mw: {:.2f}
        '''.format(self.name,
        self.start_time.date(), self.end_time.date(),
        self.min_latitude,self.max_latitude,
        self.min_longitude,self.max_longitude,
        self.min_magnitude,self.max_magnitude)
        return s

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, val):
        """
        Ensures that catalogs with formats not numpy arrray are treated as numpy.array

        Note:
            This requires that catalog classes implement the self._get_catalog_as_ndarray() function.
            This function should return structured numpy.ndarray.
            Catalog will remain None, if assigned that way in constructor.
        """
        self._catalog = val
        if self._catalog is not None:
            self._catalog = self._get_catalog_as_ndarray()
            # ensure that people are behaving, somewhat non-pythonic but needed
            if not isinstance(self._catalog, numpy.ndarray):
                raise ValueError("Error: Catalog must be numpy.ndarray! Ensure that self._get_catalog_as_ndarray()" +
                                 " returns an ndarray")

    @classmethod
    def load_catalogs(cls, filename=None, **kwargs):
        """
        Generator function to handle loading a stochastic event set.
        """
        raise NotImplementedError('load_catalogs must be overwritten in child classes')

    def write_catalog(self, binary=True):
        """
        Write catalog in bespoke format. For interoperability, CSEPCatalog classes should be used.
        But we don't want to force the user to use a CSEP catalog if they are working with their own format.
        Each model might need to implement a custom reader if the file formats are different.

        Interally, catalogs should implement some method to convert them into a pandas DataFrame.
        """
        raise NotImplementedError('write_catalog not implemented.')

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog.

        Note:
            The dataframe will be in the format of the original catalog. If you require that the
            dataframe be in the CSEP ZMAP format, you must explicitly convert the catalog.

        Returns:
            (pandas.DataFrame): This function must return a pandas DataFrame
        """
        df = pandas.DataFrame(self.catalog)
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]
        return df

    def get_number_of_events(self):
        """
        Compute the number of events from a catalog by checking its length.

        :returns: number of events in catalog, zero if catalog is None
        """
        if self.catalog is not None:
            return self.catalog.shape[0]
        else:
            return 0

    def get_epoch_times(self):
        """
        Returns the datetime of the event as the UTC epoch time (aka unix timestamp)
        """
        raise NotImplementedError('get_epoch_times() must be implemented!')

    def get_cumulative_number_of_events(self):
        """
        Returns the cumulative number of events in the catalog. Primarily used for plotting purposes.
        Defined in the base class because all catalogs should be iterable.

        Returns:
            numpy.array: numpy array of the cumulative number of events, empty array if catalog is empty.
        """
        return numpy.cumsum(numpy.ones(self.event_count))

    def get_magnitudes(self):
        """
        Extend getters to implement conversion from specific catalog type to CSEP catalog.

# use user defined stats if entered into catalog
        :returns: list of magnitudes from catalog
        """
        raise NotImplementedError('get_magnitudes must be implemented by subclasses of BaseCatalog')

    def get_datetimes(self):
        """
        Returns datetime object from timestamp representation in catalog

        :returns: list of timestamps from events in catalog.
        """
        raise NotImplementedError('get_datetimes() not implemented!')

    def get_latitudes(self):
        """
        Returns:
            (numpy.array): latitude
        """
        raise NotImplementedError('get_latitudes() not implemented!')

    def get_longitudes(self):
        """
        Returns:
            (numpy.array): longitudes
        """
        raise NotImplementedError('get_longitudes() not implemented!')

    def get_mfd(self, delta_mw=0.3, p_value=0.05):
        """
        TODO: Implement this using maximum-likelihood statistics

        Computes magnitude frequency distribution for catalog. MFD is computed by creating magnitude bins
        discretized by delta_mw.

        Requires that self.get_dataframe() is implemented in order to compute MFD.

        Args:
            delta_mw (float): Magnitude spacing for magnitude binning
            p_value (float): p_value for student's t-distribution

        Returns:
            (pandas.DataFrame): Magnitude Freq Distribution. Counts and regression statistics attached for plotting.
        """
        # getting comcat catalog as dataframe
        dm = delta_mw
        p_value = p_value
        min_mw, max_mw = self.min_magnitude, self.max_magnitude
        event_count = self.get_number_of_events()

        # pandas treats intervals as inclusive on top
        mw_inter = numpy.arange(min_mw-dm/2, max_mw+dm, dm)

        # switching into dataframe for easy manipulations
        df = self.get_dataframe()

        # get the counts in each magnitude bin
        self.mfd = pandas.DataFrame(df['counts'].groupby(pandas.cut(df['magnitude'], mw_inter)).sum())

        # cumulative counts contain the number of events greater than or equal to the magnitude
        self.mfd['counts'] = self.mfd.loc[::-1, 'counts'].cumsum()

        # get values from dataframe, might contain zeros
        vals = numpy.squeeze(self.mfd.values)
        x = numpy.array(self.mfd.index.categories.mid)

        # this is some what lenient
        if len(x) < 3:
            self.mfd['N'] = numpy.nan
            self.mfd['N_est'] = numpy.nan
            self.mfd['lower_ci'] = numpy.nan
            self.mfd['upper_ci'] = numpy.nan
            self.mfd['t_stat'] = numpy.nan
            self.mfd['a'] = numpy.nan
            self.mfd['b'] = numpy.nan
            self.mfd['ci_b'] = numpy.nan
            return self.mfd

        # this could evaluate as false if there are zeros
        N = numpy.log10(vals)
        G = numpy.vstack([numpy.ones(len(x)), x]).T

        # perform least-squares to get b-value
        # log(N) = a-bM or N = 10^a / 10^bM
        # Note: this should be maximum likelihood estimation
        a, b = numpy.linalg.lstsq(G, N, rcond=None)[0]

        # generate line to plot
        N_est = a + b*x

        # setup vars for plotting ci
        err = N - numpy.squeeze(N_est)

        n = len(x)
        t_stat = scipy.stats.t.ppf(1-p_value/2, n-2)
        mean_x = numpy.mean(x)
        se_line = numpy.sqrt(numpy.sum(numpy.power(err, 2))/(n-2))
        se_xk = numpy.sqrt(1/n+numpy.power(x-mean_x, 2)/numpy.sum(numpy.power(x-mean_x, 2)))
        confs = se_line * se_xk
        lower = N_est-t_stat*confs
        upper = N_est+t_stat*confs

        # confidence interval of b-value
        rms = numpy.sqrt(numpy.mean(numpy.power(err, 2)))
        denom = numpy.sum(numpy.power(x-mean_x, 2))
        ci_b = t_stat*rms/denom

        # wish this were functional
        self.mfd['N'] = 0.0
        self.mfd['N_est'] = 0.0
        self.mfd['lower_ci'] = 0.0
        self.mfd['upper_ci'] = 0.0
        self.mfd['t_stat'] = 0.0
        self.mfd['a'] = 0.0
        self.mfd['b'] = 0.0
        self.mfd['ci_b'] = 0.0

        # add additional state to dataframe. use .loc() indexing to ensure the data is bound to dataframe
        self.mfd.loc[self.mfd['counts'] != 0, 'N'] = N
        self.mfd.loc[self.mfd['counts'] != 0, 'N_est'] = N_est
        self.mfd.loc[self.mfd['counts'] != 0, 'lower_ci'] = lower
        self.mfd.loc[self.mfd['counts'] != 0, 'upper_ci'] = upper
        self.mfd.loc[self.mfd['counts'] != 0, 't_stat'] = t_stat
        self.mfd.loc[self.mfd['counts'] != 0, 'a'] = a
        self.mfd.loc[self.mfd['counts'] != 0, 'b'] = b
        self.mfd.loc[self.mfd['counts'] != 0, 'ci_b'] = ci_b

        # return mfd
        return self.mfd

    def filter(self, statement):
        """
        Filters the catalog based on value.

        Notes: only support lowpass, highpass style filters. Bandpass or notch not implemented yet.

        Args:
            statement (str): logical statement to evaluate, e.g., 'magnitude > 4.0'

        Returns:
            self: instance of BaseCatalog, so that this function can be chained.

        """
        operators = {'>': operator.gt,
                     '<': operator.lt,
                     '>=': operator.ge,
                     '<=': operator.le,
                     '==': operator.eq}
        name, type, value = statement.split(' ')
        idx = numpy.where(operators[type](self.catalog[name], float(value)))
        filtered = self.catalog[idx]
        self.catalog = filtered

        # update instance state before returning
        self._update_catalog_stats()

        # return self
        return self

    def _get_csep_format(self):
        """
        This method should be overwritten for catalog formats that do not adhere to the CSEP ZMAP catalog format. For
        those that do, this method will return the catalog as is.

        """
        raise NotImplementedError('_get_csep_format() not implemented.')

    def _update_catalog_stats(self):
        # update min and max values
        self.event_count = self.get_number_of_events()
        self.min_magnitude = min_or_none(self.get_magnitudes())
        self.max_magnitude = max_or_none(self.get_magnitudes())
        self.min_latitude = min_or_none(self.get_latitudes())
        self.max_latitude = max_or_none(self.get_latitudes())
        self.min_longitude = min_or_none(self.get_longitudes())
        self.max_longitude = max_or_none(self.get_longitudes())
        self.start_time = epoch_time_to_utc_datetime(min_or_none(self.get_epoch_times()))
        self.end_time = epoch_time_to_utc_datetime(max_or_none(self.get_epoch_times()))

    def _get_catalog_as_ndarray(self):
        """
        This function must be implemented if the catalog is loaded in a bespoke format.
        This function will be called anytime that a catalog is assigned
        to self.catalog and is not of type ndarray.

        The structure of the ndarray does not matter, so long as the getters can be
        implemented correctly.

        Additionally, advanced catalog operations could be carried out using GeoDataFrames and
        DataFrames.
        """
        return self.catalog


class CSEPCatalog(BaseCatalog):
    """
    Catalog stored in CSEP2 format. This catalog be used when operating within the CSEP2 software ecosystem.
    """
    # define representation for each event in catalog
    dtype = numpy.dtype([('longitude', numpy.float32),
                        ('latitude', numpy.float32),
                        ('year', numpy.int32),
                        ('month', numpy.int32),
                        ('day', numpy.int32),
                        ('magnitude', numpy.float32),
                        ('depth', numpy.float32),
                        ('hour', numpy.int32),
                        ('minute', numpy.int32),
                        ('second', numpy.int32)])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog. Explicitly casts to pandas DataFrame.

        Note:
            The dataframe will be in the format of the original catalog. If you require that the
            dataframe be in the CSEP ZMAP format, you must explicitly convert the catalog.

        Returns:
            (pandas.DataFrame): This function must return a pandas DataFrame

        Raises:
            ValueError: If self._catalog cannot be passed to pandas.DataFrame constructor, this function
                        must be overridden in the child class.
        """
        df = pandas.DataFrame(self._catalog)
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]

        if 'datetime' not in df.keys():
            df['datetime'] = self.get_datetimes()

        return df

    def get_longitudes(self):
        return self.catalog['longitude']

    def get_latitudes(self):
        return self.catalog['latitude']

    def get_magnitudes(self):
        """
        extend getters to implement conversion from specific catalog type to CSEP catalog.

        Returns:
            (numpy.array): magnitudes from catalog
        """
        return self.catalog['magnitude']

    def get_datetimes(self):
        """
        returns datetime object from timestamp representation in catalog

        :returns: list of timestamps from events in catalog.
        """
        datetimes = []
        for event in self.catalog:
            year = event['year']
            month = event['month']
            day = event['day']
            hour = event['hour']
            minute = event['minute']
            second = event['second']
            dt = datetime.datetime(year, month, day, hour=hour, minute=minute, second=second)
            datetimes.append(dt)
        return datetimes

    def _get_csep_format(self):
        return self


class UCERF3Catalog(BaseCatalog):
    """
    Handles catalog type for stochastic event sets produced by UCERF3.

    :var header_dtype: numpy.dtype description of synthetic catalog header.
    :var event_dtype: numpy.dtype description of ucerf3 catalog format
    """
    # binary format of UCERF3 catalog
    header_dtype = numpy.dtype([("file_version", ">i2"), ("catalog_size", ">i4")])
    dtype = numpy.dtype([("rupture_id", ">i4"),
                         ("parent_id", ">i4"),
                         ("generation", ">i2"),
                         ("origin_time", ">i8"),
                         ("latitude", ">f8"),
                         ("longitude", ">f8"),
                         ("depth", ">f8"),
                         ("magnitude", ">f8"),
                         ("dist_to_parent", ">f8"),
                         ("erf_index", ">i4"),
                         ("fss_index", ">i4"),
                         ("grid_node_index", ">i4")])

    def __init__(self, **kwargs):
        # initialize parent constructor
        super().__init__(**kwargs)

    @classmethod
    def load_catalogs(cls, filename=None, **kwargs):
        """
        Loads catalogs based on the merged binary file format of UCERF3. File format is described at
        https://scec.usc.edu/scecpedia/CSEP2_Storing_Stochastic_Event_Sets#Introduction.

        There is also the load_catalog method that will work on the individual binary output of the UCERF3-ETAS
        model.

        :param filename: filename of binary stochastic event set
        :type filename: string
        :returns: list of catalogs of type UCERF3Catalog
        """
        with open(filename, 'rb') as catalog_file:
            # parse 4byte header from merged file
            number_simulations_in_set = numpy.fromfile(catalog_file, dtype='>i4', count=1)[0]

            # load all catalogs from merged file
            for catalog_id in range(number_simulations_in_set):

                header = numpy.fromfile(catalog_file, dtype=cls.header_dtype, count=1)
                catalog_size = header['catalog_size'][0]

                # read catalog
                catalog = numpy.fromfile(catalog_file, dtype=cls.dtype, count=catalog_size)

                # add column that stores catalog_id in case we want to store in database
                u3_catalog = cls(filename=filename, catalog=catalog, catalog_id=catalog_id, **kwargs)

                # generator function
                yield(u3_catalog)

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog. Explicitly casts to pandas DataFrame.

        Note:
            The dataframe will be in the format of the original catalog. If you require that the
            dataframe be in the CSEP ZMAP format, you must explicitly convert the catalog.

        Returns:
            (pandas.DataFrame): This function must return a pandas DataFrame

        Raises:
            ValueError: If self._catalog cannot be passed to pandas.DataFrame constructor, this function
                        must be overridden in the child class.
        """
        df = pandas.DataFrame(self.catalog)
        # this is used for aggregrating counts
        df['counts'] = 1
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = self.catalog_id
        if 'datetime' not in df.keys():
            df['datetime'] = df['origin_time'].map(epoch_time_to_utc_datetime)
        # set index as datetime
        df.index = df['datetime']
        return df

    def get_datetimes(self):
        """
        Gets python datetime objects from time representation in catalog.

        Note:
            All times should be considered UTC time. If you are extending or working with this class make sure to
            ensure that datetime objects are not converted back to the local platform time.

        Returns:
            list: list of python datetime objects in the UTC timezone. one for each event in the catalog
        """
        datetimes = []
        for event in self.catalog:
            dt = epoch_time_to_utc_datetime(event['origin_time'])
            datetimes.append(dt)
        return datetimes

    def get_epoch_times(self):
        return self.catalog['origin_time']

    def get_magnitudes(self):
        """
        Returns array of magnitudes from the catalog.

        Returns:
            numpy.array: magnitudes of observed events in the catalog
        """
        return self.catalog['magnitude']

    def get_longitudes(self):
        return self.catalog['longitude']

    def get_latitudes(self):
        return self.catalog['latitude']

    def _get_csep_format(self):
        # TODO: possibly modify this routine to happen faster. the byteswapping is expensive.
        n = len(self.catalog)
        # allocate array for csep catalog
        csep_catalog = numpy.zeros(n, dtype=CSEPCatalog.dtype)

        for i, event in enumerate(self.catalog):
            dt = epoch_time_to_utc_datetime(event['origin_time'])
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            csep_catalog[i] = (event['longitude'].byteswap(),
                               event['latitude'].byteswap(),
                               year,
                               month,
                               day,
                               event['magnitude'].byteswap(),
                               event['depth'].byteswap(),
                               hour,
                               minute,
                               second)

        return CSEPCatalog(catalog=csep_catalog, catalog_id=self.catalog_id, filename=self.filename)


class ComcatCatalog(BaseCatalog):
    """
    Class handling retrieval of Comcat Catalogs.
    """
    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<f4'),
                         ('latitude', '<f4'),
                         ('longitude','<f4'),
                         ('depth', '<f4'),
                         ('magnitude','<f4')])

    def __init__(self, catalog_id='Comcat', format='comcat', start_epoch=None, duration_in_years=None,
                 date_accessed=None, extra_comcat_params={}, **kwargs):

        # parent class constructor
        super().__init__(catalog_id=catalog_id, format=format, **kwargs)

        self.date_accessed = date_accessed

        # if made with no catalog object, load catalog on object creation
        # TODO: Eventually, this should allow you to obtain the catalog
        #       from multiple sources. For example, the official Comcat server
        #       or from local CSEP databases.
        if self.catalog is None:
            self.start_time = self.start_time or epoch_time_to_utc_datetime(start_epoch)
            self.end_time = self.end_time or self.start_time + timedelta_from_years(duration_in_years)

            if self.start_time > self.end_time:
                raise ValueError('Error: start_time must be greater than end_time.')

            if self.start_time is None and start_epoch is None:
                    raise ValueError('Error: start_time or start_epoch must not be None.')

            if self.end_time is None and duration_in_years is None:
                raise ValueError('Error: end_time or time_delta must not be None.')

            self.load_catalog(extra_comcat_params)

    def load_catalog(self, extra_comcat_params):
        """
        Uses the libcomcat api (https://github.com/usgs/libcomcat) to parse the ComCat database for event information for
        California.

        The default parameters are given from the California testing region defined by the CSEP1 template files. starttime
        and endtime are exepcted to be datetime objects with the UTC timezone.
        Enough information needs to be provided in order to calculate a start date and end date.

        1) start_time and end_time
        2) start_time and duration_in_years
        3) epoch_time and end_time
        4) epoch_time and duration_in_years

        If start_time and start_epoch are both supplied, program will default to using start_time.
        If end_time and time_delta are both supplied, program will default to using end_time.

        This requires an internet connection and will fail if the script has no access to the server.

        Args:
            extra_comcat_params (dict): pass additional parameters to libcomcat
        """


        # get eventlist from Comcat
        eventlist = search(minmagnitude=self.min_magnitude,
            minlatitude=self.min_latitude, maxlatitude=self.max_latitude,
            minlongitude=self.min_longitude, maxlongitude=self.max_longitude,
            starttime=self.start_time, endtime=self.end_time, **extra_comcat_params)

        # eventlist is converted to ndarray in _get_catalog_as_ndarray called from setter
        self.catalog = eventlist

        # update state because we just loaded a new catalog
        self.date_accessed = datetime.datetime.utcnow()
        self._update_catalog_stats()

        return self

    def get_magnitudes(self):
        """
        Retrieves numpy.array of magnitudes from Comcat eventset.

        Returns:
            numpy.array: of magnitudes
        """
        return self.catalog['magnitude']

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog. Explicitly casts to pandas DataFrame.

        Note:
            The dataframe will be in the format of the original catalog. If you require that the
            dataframe be in the CSEP ZMAP format, you must explicitly convert the catalog. Preferentially,
            using the load_stochastic_event_set or load_catalog function defined in the base package.

        Returns:
            (pandas.DataFrame): This function must return a pandas DataFrame

        Raises:
            ValueError: If self._catalog cannot be passed to pandas.DataFrame constructor, this function
                        must be overridden in the child class.
        """
        df = pandas.DataFrame(self.catalog)
        df['counts'] = 1
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]
        if 'datetime' not in df.keys():
            df['datetime'] = df['origin_time'].map(epoch_time_to_utc_datetime)
        # set index as datetime
        df.index = df['datetime']
        return df

    def get_datetimes(self):
        """
        Returns datetime objects from catalog.

        Returns:
            (list): datetime.datetime objects
        """
        datetimes = []
        for event in self.catalog:
            datetimes.append(epoch_time_to_utc_datetime(event['origin_time']))
        return datetimes

    def get_longitudes(self):
        return self.catalog['longitude']

    def get_latitudes(self):
        return self.catalog['latitude']

    def get_epoch_times(self):
        return self.catalog['origin_time']

    def _get_catalog_as_ndarray(self):
        """
        Converts libcomcat eventlist into structured array.

        Note:
         Failure state exists if self.catalog is not bound
            to instance explicity.
        """
        # short-circuit
        if isinstance(self.catalog, numpy.ndarray):
            return self.catalog

        catalog_length = len(self.catalog)
        catalog = numpy.zeros(catalog_length, dtype=self.dtype)

        # pre-cleaned catalog is bound to self._catalog by the setter before calling this function.
        # will cause failure state if this function is called manually without binding self._catalog
        for i, event in enumerate(self.catalog):
            # debugging remove after
            catalog[i] = (event.id, datetime_to_utc_epoch(event.time),
                            event.latitude, event.longitude, event.depth, event.magnitude)

        return catalog

    def _get_csep_format(self):
        n = len(self.catalog)
        csep_catalog = numpy.zeros(n, dtype=CSEPCatalog.dtype)

        for i, event in enumerate(self.catalog):
            dt = epoch_time_to_utc_datetime(event['origin_time'])
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            csep_catalog[i] = (event['longitude'],
                               event['latitude'],
                               year,
                               month,
                               day,
                               event['magnitude'],
                               event['depth'],
                               hour,
                               minute,
                               second)

        return CSEPCatalog(catalog=csep_catalog, catalog_id=self.catalog_id, filename=self.filename)
