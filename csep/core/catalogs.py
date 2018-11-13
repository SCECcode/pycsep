import os
import numpy
import pandas
import datetime
import operator

# CSEP Imports
from csep.utils.time import epoch_time_to_utc_datetime, timedelta_from_years, datetime_to_utc_epoch


class BaseCatalog:
    """
    Base class for CSEP2 catalogs.
    """
    def __init__(self, filename=None, catalog=None, catalog_id=None, format=None, name=None):
        self.filename = filename
        self.catalog_id = catalog_id
        self.format = format
        self.name = name
        # cleans the catalog to set as ndarray
        self.catalog = catalog

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, val):
        """
        Ensures that catalogs with formats not numpy arrray are treated as numpy.array

        Note:
            This requires that catalog classes implement the self._get_catalog_as_ndarray() function.
            This function should structured numpy.ndarray.
        """
        self._catalog = val
        if self._catalog is not None:
            if not isinstance(self._catalog, numpy.ndarray):
                self._catalog = self._get_catalog_as_ndarray()

    @classmethod
    def load_catalog(self):
        # TODO: Make classmethod and remove from constructor. Classes should be loaded through factory function.
        """
        Must be implemented for each model that gets used within CSEP.
        Base class assumes that catalogs are stored in default format which is defined.
        """
        raise NotImplementedError('load_catalog not implemented.')

    @classmethod
    def load_catalogs(cls, filename=None, **kwargs):
        """
        Generator function to handle loading a stochastic event set.
        """
        raise NotImplementedError('load_catalogs must be overwritten in child classes')

    def _get_csep_format(self):
        """
        This method should be overwritten for catalog formats that do not adhere to the CSEP ZMAP catalog format. For
        those that do, this method will return the catalog as is.

        """
        raise NotImplementedError('_get_as_csep_format not implemented.')

    def write_catalog(self, binary = True):
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
            return len(self.catalog)
        else:
            return 0

    def get_cumulative_number_of_events(self):
        """
        Returns the cumulative number of events in the catalog. Primarily used for plotting purposes.
        Defined in the base class because all catalogs should be iterable.

        Returns:
            numpy.array: numpy array of the cumulative number of events, empty array if catalog is empty.
        """
        num_events = self.get_number_of_events()
        return numpy.cumsum(numpy.ones(num_events))

    def get_magnitudes(self):
        """
        Extend getters to implement conversion from specific catalog type to CSEP catalog.

        :returns: list of magnitudes from catalog
        """
        raise NotImplementedError

    def get_datetimes(self):
        """
        Returns datetime object from timestamp representation in catalog

        :returns: list of timestamps from events in catalog.
        """
        raise NotImplementedError('get_datetimes not implemented.')

    def get_latitudes_and_longitudes(self):
        """
        Returns:
            list of tuples (latitude, longitude)
        """
        raise NotImplementedError('get_latitudes_and_longitudes not implemented.')

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
        return self


class CSEPCatalog(BaseCatalog):
    """
    Catalog stored in CSEP2 format. This catalog be used when operating within the CSEP2 software ecosystem.
    """
    # define representation for each event in catalog
    csep_dtype = [('longitude', numpy.float32),
                       ('latitude', numpy.float32),
                       ('year', numpy.int32),
                       ('month', numpy.int32),
                       ('day', numpy.int32),
                       ('magnitude', numpy.float32),
                       ('depth', numpy.float32),
                       ('hour', numpy.int32),
                       ('minute', numpy.int32),
                       ('second', numpy.int32)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write_catalog(self, binary = True):
        """
        Write catalog in zipped ZMAP format. For interoperability all files should write to the same format.
        Each model might need to implement a custom reader if the file formats are different.

        1. Longitude [deg]
        2. Latitude [deg]
        3. Year [e.g., 2005]
        4. Month
        5. Day
        6. Magnitude
        7. Depth [km]
        8. Hour
        9. Minute
        10. Second

        Catalog will be represented as a strucutred Numpy array with could be cast
        into pandas DataFrame.
        """
        raise NotImplementedError

    def _get_csep_format(self):
        return self

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


class UCERF3Catalog(BaseCatalog):
    """
    Handles catalog type for stochastic event sets produced by UCERF3.

    :var header_dtype: numpy.dtype description of synthetic catalog header.
    :var event_dtype: numpy.dtype description of ucerf3 catalog format
    """
    # binary format of UCERF3 catalog
    header_dtype = numpy.dtype([("file_version", ">i2"), ("catalog_size", ">i4")])
    event_dtype = numpy.dtype([
        ("rupture_id", ">i4"),
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
        ("grid_node_index", ">i4")
    ])

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
        catalogs = []
        with open(filename, 'rb') as catalog_file:
            # parse 4byte header from merged file
            number_simulations_in_set = numpy.fromfile(catalog_file, dtype='>i4', count=1)[0]

            # load all catalogs from merged file
            for catalog_id in range(number_simulations_in_set):

                header = numpy.fromfile(catalog_file, dtype=cls.header_dtype, count=1)
                catalog_size = header['catalog_size'][0]

                # read catalog
                catalog = numpy.fromfile(catalog_file, dtype=cls.event_dtype, count=catalog_size)

                # add column that stores catalog_id in case we want to store in database
                u3_catalog = cls(filename=filename, catalog=catalog, catalog_id=catalog_id, **kwargs)

                # generator function
                yield(u3_catalog)

    def _get_csep_format(self):
        # TODO: possibly modify this routine to happen faster. the byteswapping is expensive.
        n = len(self.catalog)
        # allocate array for csep catalog
        csep_catalog = numpy.zeros(n, dtype=CSEPCatalog.csep_dtype)

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
        print('took {} seconds to convert to csep format.'.format(t1-t0))

        return CSEPCatalog(catalog=csep_catalog, catalog_id=self.catalog_id, filename=self.filename)

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
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]

        if 'datetime' not in df.keys():
            df['datetime'] = self.get_datetimes()

        return df

    def convert_to_csep_format(self):
        """
        Function will convert native data frame format into CSEP ZMAP catalog format.

        Returns:
            (:class:`~csep.core.catalogs.CSEPCatalog`): instance of CSEPCatalog
        """
        raise NotImplementedError("convert_to_csep_format not yet implemented")

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

    def get_magnitudes(self):
        """
        Returns array of magnitudes from the catalog.

        Returns:
            numpy.array: magnitudes of observed events in the catalog
        """
        return self.catalog['magnitude']


class ComcatCatalog(BaseCatalog):
    """
    Class handling retrieval of Comcat Catalogs.
    """
    comcat_dtype = numpy.dtype([('id', 'S256'), ('epoch_time', '<f4'), ('latitude', '<f4'), ('longitude','<f4'),
                            ('depth', '<f4'), ('magnitude','<f4')])

    def __init__(self, catalog_id='Comcat', format='Comcat', min_magnitude=2.55,
            min_latitude=31.50, max_latitude=43.00,
            min_longitude=-125.40, max_longitude=-113.10,
            limit=20000, start_time=None, end_time=None,
            start_epoch=None, duration_in_years=None, extra_comcat_params={},
            **kwargs):

        self.min_magnitude = min_magnitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude

        # parent class constructor
        super().__init__(**kwargs)

        if start_time is None and start_epoch is None:
                raise ValueError('Error: start_time or start_epoch must not be None.')

        if end_time is None and duration_in_years is None:
            raise ValueError('Error: end_time or time_delta must not be None.')

        self.start_time = start_time or epoch_time_to_utc_datetime(start_epoch)
        self.end_time = end_time or self.start_time + timedelta_from_years(duration_in_years)
        if self.start_time > self.end_time:
            raise ValueError('Error: start_time must be greater than end_time.')

        # load catalog on object creation
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
        """
        from libcomcat.search import search

        # get eventlist from Comcat
        eventlist = search(minmagnitude=self.min_magnitude,
            minlatitude=self.min_latitude, maxlatitude=self.max_latitude,
            minlongitude=self.min_longitude, maxlongitude=self.max_longitude,
            starttime=self.start_time, endtime=self.end_time, **extra_comcat_params)

        # eventlist is converted to ndarray in _get_catalog_as_ndarray called from setter
        self.catalog = eventlist

        return self

    def get_magnitudes(self):
        """
        Retrieves numpy.array of magnitudes from Comcat eventset.

        Returns:
            numpy.array: of magnitudes
        """
        magnitudes = []
        for event in self.catalog:
            magnitudes.append(event['magnitude'])
        return numpy.array(magnitudes)

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
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]

        if 'datetime' not in df.keys():
            df['datetime'] = self.get_datetimes()

        return df

    def get_datetimes(self):
        """
        Returns datetime objects from catalog.

        Returns:
            (list): datetime.datetime objects
        """
        datetimes = []
        for event in self.catalog:
            datetimes.append(epoch_time_to_utc_datetime(event['epoch_time']))
        return datetimes

    def _get_catalog_as_ndarray(self):
        """
        Converts libcomcat eventlist into structured array.

        Note:
            Be careful calling this function. Failure state exists if self.catalog is not bound
            to instance explicity.
        """
        events = []
        catalog_length = len(self.catalog)
        catalog = numpy.zeros(catalog_length, dtype=self.comcat_dtype)

        # pre-cleaned catalog is bound to self._catalog by the setter before calling this function.
        # will cause failure state if this function is called manually without binding self._catalog
        for i, event in enumerate(self.catalog):
            catalog[i] = (event.id, datetime_to_utc_epoch(event.time),
                            event.latitude, event.longitude, event.depth, event.magnitude)

        return catalog

    def _get_csep_format(self):
        n = len(self.catalog)
        csep_catalog = numpy.zeros(n, dtype=CSEPCatalog.csep_dtype)

        for i, event in enumerate(self.catalog):
            dt = epoch_time_to_utc_datetime(event['epoch_time'])
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
