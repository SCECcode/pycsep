import os
import numpy
import pandas
import datetime

# CSEP Imports
from csep.utils.time import epoch_time_to_utc_datetime, timedelta_from_years


class BaseCatalog:
    """
    Base class for CSEP2 catalogs.
    """
    def __init__(self, filename=None, catalog=None, catalog_id=None, format=None, name=None, lazy_load=True):
        self.filename = filename
        self.catalog_id = catalog_id
        self.catalog = catalog
        self.format = format
        self.name = name

        # we might not want to defer loading of catalog until it is actually needed
        # by analysis routines as catalogs can be quite large
        if not lazy_load and self.catalog is None:
            self.load_catalog()

    def load_catalog(self, merged=False):
        """
        Must be implemented for each model that gets used within CSEP.
        Base class assumes that catalogs are stored in default format which is defined.

        :param merged: if each catalog is specified as an individual file or merged
        :type merged: bool
        """
        raise NotImplementedError('load_catalog not implemented.')

    def convert_to_csep_format(self):
        """
        This method should be overwritten for catalog formats that do not adhere to the CSEP ZMAP catalog format. For
        those that do, this method will return the catalog as is.

        """
        raise NotImplementedError('convert_to_csep_format not implemented.')

    def write_catalog(self, binary = True):
        """
        Write catalog in zipped ZMAP format. For interoperability all files should write to the same format.
        Each model might need to implement a custom reader if the file formats are different.

        Catalog will be represented as a strucutred Numpy array with could be cast
        into pandas DataFrame.
        """
        raise NotImplementedError('write_catalog not implemented.')

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog. Explicitly casts to pandas DataFrame.

        Note:
            The dataframe will be in the format of the original catalog. If you require that the
            dataframe be in the CSEP ZMAP format, you must explicitly convert the catalog.
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
        Defined in the base class because all functions should be iterable.

        Note:
        Returns:
            numpy.array: numpy array of the cumulative number of events, empty array if catalog is empty.
        """
        num_events = self.get_number_of_events()
        return numpy.cumsum(numpy.ones(num_events))



    def get_magnitudes(self):
        """
        extend getters to implement conversion from specific catalog type to CSEP catalog.

        :returns: list of magnitudes from catalog
        """
        raise NotImplementedError

    def get_datetimes(self):
        """
        returns datetime object from timestamp representation in catalog

        :returns: list of timestamps from events in catalog.
        """
        raise NotImplementedError('get_datetimes not implemented.')

    def get_latitudes_and_longitudes(self):
        """
        Returns:
            list of tuples (latitude, longitude)
        """
        raise NotImplementedError('get_latitudes_and_longitudes not implemented.')


class CSEPCatalog(BaseCatalog):
    """
    Catalog stored in CSEP2 format. This catalog will have the most functionality and should be used
    when operating within the CSEP2 software ecosystem.
    """
    def __init__(self, filename=None, catalog=None, catalog_id=None, format='CSEP', lazy_load=True):
        self.filename = filename
        self.catalog_id = catalog_id
        self.catalog = catalog

        # define representation for each event in catalog
        self.csep_dtype = [('longitude', numpy.float32),
                           ('latitude', numpy.float32),
                           ('year', numpy.int32),
                           ('month', numpy.int32),
                           ('day', numpy.int32),
                           ('magnitude', numpy.float32),
                           ('depth', numpy.float32),
                           ('hour', numpy.int32),
                           ('minute', numpy.int32),
                           ('second', numpy.int32)]

    def load_catalog(self, merged=False):
        """
        Must be implemented for each model that gets used within CSEP.
        Base class assumes that catalogs are stored in default format which is defined.

        :param merged: if each catalog is specified as an individual file or merged
        :type merged: bool
        """
        raise NotImplementedError

    def convert_to_csep_format(self):
        """
        This method should be overwritten for catalog formats that do not adhere to the CSEP ZMAP catalog format. For
        those that do, this method will return the catalog as is.

        """
        return self.get_dataframe()

    def write_csep_catalog(self, binary = True):
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

    def get_dataframe(self):
        """
        Returns pandas Dataframe describing the catalog. Explicitly casts to pandas DataFrame.
        """
        df = pandas.DataFrame(self.catalog)
        if 'catalog_id' not in df.keys():
            df['catalog_id'] = [self.catalog_id for _ in range(len(self.catalog))]
        return df

    def get_magnitudes(self):
        """
        extend getters to implement conversion from specific catalog type to CSEP catalog.

        :returns: list of magnitudes from catalog
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

    def _get_latitude_and_longitude(self):
        location = []
        for event in self.catalog:
            location.append()

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
    def load_catalogs(cls, filename=None, convert=False, as_dataframe=False, **kwargs):
        """
        Loads catalogs based on the merged binary file format of UCERF3. File format is described at
        https://scec.usc.edu/scecpedia/CSEP2_Storing_Stochastic_Event_Sets#Introduction.

        There is also the load_catalog method that will work on the individual binary output of the UCERF3-ETAS
        model.

        :param filename: filename of binary stochastic event set
        :type filename: string
        :param convert: whether or not to convert catalog into CSEP format.
        :type convert: bool
        :param as_dataframe: if true, store in class as dataframe. if false, store as numpy array
        :type as_dataframe: bool
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

                # default format is numpy.array()
                if as_dataframe:
                    catalog = pandas.DataFrame(catalog)
                    catalog['catalog_id'] = [catalog_id for catalog_id in range(catalog_size)]

                # add column that stores catalog_id in case we want to store in database
                u3_catalog = cls(filename=filename, catalog=catalog, catalog_id=catalog_id, **kwargs)

                # generator function
                yield(u3_catalog)

    def convert_to_csep_format(self):
        """
        Function will convert native data frame format into CSEP ZMAP catalog format.
        :returns: instance of CSEPCatalog
        """
        raise NotImplementedError("convert_to_csep_format not yet implemented")

    def load_catalog(self):
        """
        Loads single catalog from UCERF3 stochastic event set.
        """
        raise NotImplementedError("load_catalog not yet implemented.")

    def get_datetimes(self):
        """
        Gets python datetime objects from time representation in catalog.

        Note:
            All times should be considered UTC time. If you are extending or working with this class make sure to
            ensure that datetime objects are not converted back to the local platform time.

        Returns:
            numpy.array: list of python datetime objects in the UTC timezone. one for each event in the catalog
        """
        datetimes = []
        for event in self.catalog:
            dt = epoch_time_to_utc_datetime(event['origin_time'])
            datetimes.append(dt)
        return numpy.array(datetimes)


class ComcatCatalog(BaseCatalog):
    """
    Class handling retrieval of Comcat Catalogs.
    """

    def __init__(self, min_magnitude=2.55,
            min_latitude=31.50, max_latitude=43.00,
            min_longitude=-125.40, max_longitude=-113.10,
            limit=20000, start_time=None, end_time=None,
            start_epoch=None, duration_in_years=None, catalog_id='Comcat', **kwargs):
        """
        The default parameters are given from the California testing region defined by the CSEP1 template files. starttime
        and endtime are exepcted to be datetime objects with the UTC timezone.
        Enough information needs to be provided in order to calculate a start date and end date.

        1) start_time and end_time
        2) start_time and duration_in_years
        3) epoch_time and end_time
        4) epoch_time and duration_in_years

        If start_time and start_epoch are both supplied, program will default to using start_time.
        If end_time and time_delta are both supplied, program will default to using end_time.

        :param min_magnitude: minimum magnitude query
        :type min_magnitude: float
        :param min_latitude: min latitude of bounding box
        :type min_latitude: float
        :param max_latitude: max latitude of bounding box
        :type min_latitude: float
        :param min_longitude: min longitude of bounding box
        :type min_longitude: float
        :param max_longitude: max longitude of bounding box
        :type max_longitude: float
        :param limit:
        :param start_time:
        :type start_time:
        :param end_time:
        :type end_time:
        :param duration_in_years:
        :type duration_in_years:
        """
        self.min_magnitude = min_magnitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.catalog_id = catalog_id

        if start_time is None and start_epoch is None:
            raise ValueError('Error: start_time or start_epoch must not be None.')

        if end_time is None and duration_in_years is None:
            raise ValueError('Error: end_time or time_delta must not be None.')

        self.start_time = start_time or epoch_time_to_utc_datetime(start_epoch)
        self.end_time = end_time or self.start_time + timedelta_from_years(duration_in_years)
        if self.start_time > self.end_time:
            raise ValueError('Error: start_time must be greater than end_time.')

        super().__init__(**kwargs)

    def load_catalog(self, **kwargs):
        """
        Uses the libcomcat api (https://github.com/usgs/libcomcat) to parse the ComCat database for event information for
        California.

        This requires an internet connection and will fail if the script has no access to the server.
        """
        from libcomcat.search import search

        # get eventlist from Comcat
        eventlist = search(minmagnitude=self.min_magnitude,
            minlatitude=self.min_latitude, maxlatitude=self.max_latitude,
            minlongitude=self.min_longitude, maxlongitude=self.max_longitude,
            starttime=self.start_time, endtime=self.end_time, **kwargs)

        self.catalog = eventlist
        return self.catalog

    def get_datetimes(self):
        datetimes = []
        for event in self.catalog:
            datetimes.append(event.time)
        return numpy.array(datetimes)
