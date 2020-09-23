import csv
import json
import operator
import os
import datetime

# 3rd party required for core package
import numpy
import pandas

# CSEP Imports
from csep.core import regions
from csep.utils.time_utils import epoch_time_to_utc_datetime, datetime_to_utc_epoch, strptime_to_utc_datetime, \
    millis_to_days, parse_string_format, days_to_millis, strptime_to_utc_epoch, utc_now_datetime, create_utc_datetime
from csep.utils.stats import min_or_none, max_or_none
from csep.utils.calc import discretize
from csep.utils.comcat import SummaryEvent
from csep.core.exceptions import CSEPSchedulerException, CSEPCatalogException, CSEPIOException
from csep.utils.calc import bin1d_vec
from csep.utils.constants import CSEP_MW_BINS
from csep.utils.log import LoggingMixin
from csep.utils.readers import csep_ascii


class AbstractBaseCatalog(LoggingMixin):
    """
    Abstract catalog base class for PyCSEP catalogs. This class should not and cannot be used on its own. This just
    provides the interface for implementing custom catalog classes.

    """

    dtype = None

    def __init__(self, filename=None, data=None, catalog_id=None, format=None, name=None, region=None,
                 compute_stats=True, filters=None, metadata=None, date_accessed=None):

        """ Standard catalog format for CSEP catalogs. Primary event data are stored in structured numpy array. Additional
            metadata are available by the event_id in the catalog metadata information.

        Args:
            filename: location of catalog
            catalog (numpy.ndarray or eventlist): catalog data
            catalog_id: catalog id number (used for stochastic event set forecasts)
            format: identification used for serialization
            name: human readable name of catalog
            region: spatial and magnitude region
            compute_stats: whether statistics should be computed for the catalog
            filters (str or list): filtering operations to apply to the catalog
            metadata (dict): additional information for events
            date_accessed (str): time string when catalog was accessed
        """
        super().__init__()
        self.filename = filename
        self.catalog_id = catalog_id
        self.format = format
        self.name = name
        self.region = region
        self.compute_stats = compute_stats
        self.filters = filters or []
        self.date_accessed = date_accessed or utc_now_datetime()  # type datetime.datetime

        # used to store additional event information based on the event_id key, if no event_id will default to an
        # integer index
        self.metadata = metadata or {}

        # cleans the catalog to set as ndarray, see setter.
        self.catalog = data  # type: numpy.ndarray

        # use user defined stats if entered into catalog
        if data is not None and self.compute_stats:
            self.update_catalog_stats()

    def __eq__(self, other):
        """ Compares whether two catalogs are equal by comparing their dicts. """
        return self.to_dict() == other.to_dict()

    def __str__(self):
        self.update_catalog_stats()

        s = f'''
        Name: {self.name}
        
        Start Date: {self.start_time}
        End Date: {self.end_time}
        
        Latitude: ({self.min_latitude}, {self.max_latitude})
        Longitude: ({self.min_longitude}, {self.max_longitude})
        
        Min Mw: {self.min_magnitude}
        Max Mw: {self.max_magnitude}
        
        Event Count: {self.event_count}
        '''
        return s

    def to_dict(self):
        """
        Serializes class to json dictionary.

        Returns:
            catalog as dict

        """
        excluded = ['_catalog']
        out = {}
        for k, v in self.__dict__.items():
            # note: if 'v' is callable that implies that we have a function bound to a class-member. this happens
            # for the catalog forecast and requires excluding this value.
            if not callable(v) and k not in excluded:
                if hasattr(v, 'to_dict'):
                    new_v = v.to_dict()
                else:
                    new_v = v
                if k.startswith('_'):
                    out[k[1:]] = new_v
                else:
                    out[k] = new_v
        out['catalog'] = []
        for line in list(self.catalog.tolist()):
            new_line = []
            for item in line:
                # try to decode, if it fails just use original, we use this to handle string-based event_ids
                try:
                    item = item.decode('utf-8')
                except:
                    pass
                finally:
                    new_line.append(item)
            out['catalog'].append(new_line)
        return out

    @property
    def event_count(self):
        """ Number of events in catalog """
        return self.get_number_of_events()

    @classmethod
    def load_catalog(cls, filename, loader=csep_ascii, **kwargs):
        raise NotImplementedError("subclass should implement load_catalog funtion.")

    @classmethod
    def from_dict(cls, adict, **kwargs):
        """ Creates a class from the dictionary representation of the class state. The catalog is serialized into a list of
        tuples that contain the event information in the order defined by the dtype.

        This needs to handle reading in region information at some point.
        """

        # could these be class values? can be changed later.
        exclude = ['_catalog']
        time_members = ['date_accessed', 'start_time', 'end_time']
        catalog = adict.get('catalog', None)

        out = cls(data=catalog, **kwargs)

        for k, v in out.__dict__.items():
            if k not in exclude:
                if k not in time_members:
                    try:
                        setattr(out, k, adict[k])
                    except KeyError:
                        pass
                else:
                    setattr(out, k, _none_or_datetime(adict[k]))
        return out

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        """
        Creates catalog from dataframe. Dataframe must have columns that are equivalent to whatever fields
        the catalog expects in the catalog dtype.

        For example:

                cat = CSEPCatalog()
                df = cat.get_dataframe()
                new_cat = CSEPCatalog.from_dataframe(df)

        Args:
            df (pandas.DataFrame): pandas dataframe
            **kwargs:

        Returns:
            Catalog

        """
        catalog_id = None
        try:
            catalog_id = df['catalog_id'].iloc[0]
        except KeyError:
            pass
        col_list = list(cls.dtype.names)
        # we want this to be a structured array not a record array and only returns core attributes listed in dtype
        # loses information about the region and event meta data
        catalog = numpy.ascontiguousarray(df[col_list].to_records(index=False), dtype=cls.dtype)
        out_cls = cls(data=catalog, catalog_id=catalog_id, **kwargs)
        return out_cls

    @classmethod
    def load_json(cls, filename, **kwargs):
        """ Loads catalog from JSON file """
        with open(filename, 'r') as f:
            adict = json.load(f)
            return cls.from_dict(adict, **kwargs)

    def write_json(self, filename):
        """ Writes catalog to json file

        Args:
            filename (str): path to save file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, separators=(',', ': '), sort_keys=True, default=str)

    @property
    def catalog(self):
        return self._catalog

    @property
    def data(self):
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
        if self.compute_stats and self._catalog is not None:
            self.update_catalog_stats()

    def _get_catalog_as_ndarray(self):
        """
        This function will be called anytime that a catalog is assigned to self.catalog

        The purpose of this function is to ensure that the catalog is being properly parsed into the correct format, and
        to prevent users of the catalog classes from assigning improper data types.

        This also acts as a convenience to allow easy assignment of different types to the catalog. The default
        implementation of this function expects that the data are arranged as a collection of tuples corresponding to
        the catalog data type.
        """
        """
        Converts eventlist into ndarray format.

        Note:
         Failure state exists if self.catalog is not bound
            to instance explicity.
        """
        # short-circuit
        if isinstance(self.catalog, numpy.ndarray):
            return self.catalog
        # if catalog is not a numpy array, class must have dtype information
        catalog_length = len(self.catalog)
        catalog = numpy.empty(catalog_length, dtype=self.dtype)
        if catalog_length == 0:
            return catalog
        if isinstance(self.catalog[0], (list, tuple)):
            for i, event in enumerate(self.catalog):
                catalog[i] = tuple(event)
        elif isinstance(self.catalog[0], SummaryEvent):
            for i, event in enumerate(self.catalog):
                catalog[i] = (event.id, datetime_to_utc_epoch(event.time),
                              event.latitude, event.longitude, event.depth, event.magnitude)
        else:
            raise TypeError("Catalog data must be list of events tuples with order:\n"
                            f"{', '.join(self.dtype.names)} or \n"
                            "list of SummaryEvent type.")
        return catalog

    def write_ascii(self, filename, write_header=True, write_empty=True, append=False, id_col='id'):
        """
        Write catalog in csep2 ascii format.

        This format only uses the required variables from the catalog and should work by default. It can be overwritten
        if an event_id (or other columns should be used). By default, the routine will look for a column the catalog array
        called 'id' and will populate the event_id column with these values. If the 'id' column is not found, then it will
        leave this column blank

        Short format description (comma separated values):
            longitude, latitude, M, time_string format="%Y-%m-%dT%H:%M:%S.%f", depth, catalog_id, [event_id]

        Args:
            filename (str): the file location to write the the ascii catalog file
            write_header (bool): Write header string (default true)
            write_empty (bool): Write file event if no events in catalog
            append (bool): If true, append to the filename
            id_col (str): name of event_id column (if included)

        Returns:
            NoneType
        """
        # longitude, latitude, M, epoch_time (time in millisecond since Unix epoch in GMT), depth, catalog_id, event_id
        header = ['lon', 'lat', 'mag', 'time_string', 'depth', 'catalog_id', 'event_id']
        if append:
            write_string = 'a'
        else:
            write_string = 'w'
        with open(filename, write_string) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=header, delimiter=',')
            if write_header:
                writer.writeheader()
                if write_empty and self.event_count == 0:
                    return
            # create iterator from catalog columns
            try:
                event_ids = self.catalog[id_col]
            except ValueError:
                event_ids = [''] * self.event_count
            row_iter = zip(self.get_longitudes(),
                           self.get_latitudes(),
                           self.get_magnitudes(),
                           self.get_epoch_times(),
                           self.get_depths(),
                           # populate list with `self.event_count` elements with val self.catalog_id
                           [self.catalog_id] * self.event_count,
                           event_ids)
            # write csv file using DictWriter interface
            for row in row_iter:
                try:
                    event_id = row[6].decode('utf-8')
                except AttributeError:
                    event_id = row[6]
                # create dictionary for each row
                adict = {'lon': row[0],
                         'lat': row[1],
                         'mag': row[2],
                         'time_string': str(epoch_time_to_utc_datetime(row[3]).replace(tzinfo=None)).replace(' ', 'T'),
                         'depth': row[4],
                         'catalog_id': row[5],
                         'event_id': event_id}
                writer.writerow(adict)

    def to_dataframe(self, with_datetime=False):
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
        df['counts'] = 1
        df['catalog_id'] = self.catalog_id
        if with_datetime:
            df['datetime'] = df['origin_time'].map(epoch_time_to_utc_datetime)
            df.index = df['datetime']
        # queries the region for the index of each event
        if self.region is not None:
            df['region_id'] = self.region.get_index_of(self.get_longitudes(), self.get_latitudes())
            try:
                # bin magnitudes
                df['mag_id'] = self.get_mag_idx()
            except AttributeError:
                pass
        # set index as datetime
        return df

    def get_mag_idx(self):
        """ Return magnitude index from region magnitudes """
        try:
            return bin1d_vec(self.get_magnitudes(), self.region.magnitudes, right_continuous=True)
        except AttributeError:
            raise CSEPCatalogException("Cannot return magnitude index without self.region.magnitudes")

    def get_spatial_idx(self):
        """ Return spatial index of region for a longitudes and latitudes in catalog. """
        try:
            region_idx = self.region.get_index_of(self.get_longitudes(), self.get_latitudes())
        except AttributeError:
            raise CSEPCatalogException("Must have region information to compute region index.")
        return region_idx

    def get_event_ids(self):
        return self.catalog['id']

    def get_number_of_events(self):
        """
        Computes the number of events from a catalog by checking its length.

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
        return self.catalog['origin_time']

    def get_cumulative_number_of_events(self):
        """
        Returns the cumulative number of events in the catalog.

        Primarily used for plotting purposes.

        Returns:
            numpy.array: numpy array of the cumulative number of events, empty array if catalog is empty.
        """
        return numpy.cumsum(numpy.ones(self.event_count))

    def get_magnitudes(self):
        """
        Returns magnitudes of all events in catalog

        """
        return self.catalog['magnitude']

    def get_datetimes(self):
        """
        Returns datetime object from timestamp representation in catalog

        :returns: list of timestamps from events in catalog.
        """
        """ Returns datetimes from events in catalog """
        return list(map(epoch_time_to_utc_datetime, self.get_epoch_times()))

    def get_latitudes(self):
        """
        Returns latitudes of all events in catalog
        """
        return self.catalog['latitude']

    def get_longitudes(self):
        """
        Returns longitudes of all events in catalog

        Returns:
            (numpy.array): longitudes
        """
        return self.catalog['longitude']

    def get_depths(self):
        """ Returns depths of all events in catalog """
        return self.catalog['depth']

    def filter(self, statements=None, in_place=True):
        """
        Filters the catalog based on statements. This function takes about 60% of the run-time for processing UCERF3-ETAS
        simulations, so likely all other simulations. Implementations should try and limit how often this function
        will be called.

        Args:
            statements (str, iter): logical statements to evaluate, e.g., ['magnitude > 4.0', 'year >= 1995']
            in_place (bool): return new instance of catalog

        Returns:
            self: instance of AbstractBaseCatalog, so that this function can be chained.

        """
        if not self.filters and statements is None:
            raise CSEPCatalogException("Must provide filter statements to function or class to filter")

        def parse_datetime_to_origin_time(dt_input):
            """ Parses datetime strings with various formats. Handles datetime objects and datetime strings.
            If datetime object is not time aware will assume that time is UTC, if it is timezone aware will convert to UTC timezone. """
            if type(dt_input) == datetime.datetime:
                # check for time zone
                if dt_input.tzinfo == None:
                    return create_utc_datetime(dt_input)
            # handle it as string
            try:
                format = parse_string_format(dt_input)
            except:
                raise CSEPIOException(
                    "Supported time-string formats are '%Y-%m-%dT%H:%M:%S.%f' and '%Y-%m-%dT%H:%M:%S'")
            return strptime_to_utc_datetime(dt_input, format)

        # programmatically assign operators
        operators = {'>': operator.gt,
                     '<': operator.lt,
                     '>=': operator.ge,
                     '<=': operator.le,
                     '==': operator.eq}

        # filter catalogs, implied logical and
        if statements is None:
            statements = self.filters

        if isinstance(statements, str):
            name = statements.split(' ')[0]
            if name == 'datetime':
                name, oper, date, time = statements.split(' ')
                # can be a datetime.datetime object or datetime string, if we want to support filtering on meta data it
                # can happen here. but need to determine what to do if entry are not present bc meta data does not
                # need to be square
                value = strptime_to_utc_epoch(' '.join([date, time]))
                idx = numpy.where(operators[oper](self.get_epoch_times(), value))
                filtered = self.catalog[idx]
            else:
                name, oper, value = statements.split(' ')
                filtered = self.catalog[operators[oper](self.catalog[name], float(value))]
        elif isinstance(statements, (list, tuple)):
            # slower but at the convenience of not having to call multiple times
            filters = list(statements)
            filtered = numpy.copy(self.catalog)
            for filt in filters:
                name = filt.split(' ')[0]
                if name == 'datetime':
                    name, oper, date, time = filt.split(' ')
                    # can be a datetime.datetime object or datetime string, if we want to support filtering on meta data it
                    # can happen here. but need to determine what to do if entry are not present bc meta data does not
                    # need to be square
                    value = strptime_to_utc_epoch(' '.join([date, time]))
                    idx = numpy.where(operators[oper](self.get_epoch_times(), value))
                    filtered = self.catalog[idx]
                else:
                    name, oper, value = filt.split(' ')
                    filtered = filtered[operators[oper](filtered[name], float(value))]
        else:
            raise ValueError('statements should be either a string or list or tuple of strings')
        # can return new instance of class or original instance
        self.filters = statements
        if in_place:
            self.catalog = filtered
            return self
        else:
            # make and return new object
            cls = self.__class__
            inst = cls(data=filtered, catalog_id=self.catalog_id, format=self.format, name=self.name,
                       region=self.region)
            return inst

    def filter_spatial(self, region=None, update_stats=False):
        """
        Removes events outside of the region. This takes some time and should be used sparingly. Typically for isolating a region
        near the mainshock or inside a testing region. This should not be used to create gridded style data sets.

        Args:
            region: csep.utils.spatial.Region

        Returns:
            self

        """
        if region is None and self.region is None:
            raise CSEPCatalogException("region provided to function or bound to the catalog instance.")

        # update the region to the new region
        if region is not None:
            self.region = region

        mask = self.region.get_masked(self.get_longitudes(), self.get_latitudes())
        # logical index uses opposite boolean values than masked arrays.
        filtered = self.catalog[~mask]
        self.catalog = filtered

        if update_stats:
            self.update_catalog_stats()
        return self

    def apply_mct(self, m_main, event_epoch, mc=2.5):
        """
        Applies time-dependent magnitude of completeness following a mainshock. Taken
        from Eq. (15) from Helmstetter et al., 2006.

        Args:
            m_main (float): mainshock magnitude
            event_epoch: epoch time in millis of event
            mc (float): mag_completeness

        Returns:
        """

        def compute_mct(t, m):
            return m - 4.5 - 0.75 * numpy.log10(t)

        # compute critical time for efficiency
        t_crit_days = 10 ** -((mc - m_main + 4.5) / 0.75)
        t_crit_millis = days_to_millis(t_crit_days)

        times = self.get_epoch_times()
        mws = self.get_magnitudes()

        # catalogs are stored stored by milliseconds
        t_crit_epoch = t_crit_millis + event_epoch

        # another short-circuit, again assumes that catalogs are sorted in time
        if times[0] > t_crit_epoch:
            return self

        # this is used to index the array, starting with accepting all events
        filter = numpy.ones(self.event_count, dtype=numpy.bool)
        for i, (mw, time) in enumerate(zip(mws, times)):
            # we can break bc events are sorted in time
            if time > t_crit_epoch:
                break
            if time < event_epoch:
                continue
            time_from_mshock_in_days = millis_to_days(time - event_epoch)
            mct = compute_mct(time_from_mshock_in_days, m_main)
            # ignore events with mw < mct
            if mw < mct:
                filter[i] = False

        filtered = self.catalog[filter]
        self.catalog = filtered
        return self

    def get_csep_format(self):
        """
        This method should be overwritten for catalog formats that do not adhere to the CSEP ZMAP catalog format. For
        those that do, this method will return the catalog as is.

        """
        raise NotImplementedError('get_csep_format() not implemented.')

    def update_catalog_stats(self):
        """ Compute summary statistics of events in catalog """
        # update min and max values
        self.min_magnitude = min_or_none(self.get_magnitudes())
        self.max_magnitude = max_or_none(self.get_magnitudes())
        self.min_latitude = min_or_none(self.get_latitudes())
        self.max_latitude = max_or_none(self.get_latitudes())
        self.min_longitude = min_or_none(self.get_longitudes())
        self.max_longitude = max_or_none(self.get_longitudes())
        self.start_time = epoch_time_to_utc_datetime(min_or_none(self.get_epoch_times()))
        self.end_time = epoch_time_to_utc_datetime(max_or_none(self.get_epoch_times()))

    def spatial_counts(self):
        """
        Returns counts of events within discrete spatial region

        We figure out the index of the polygons and create a map that relates the spatial coordinate in the
        Cartesian grid with with the polygon in region.

        Returns:
            ndarray containing the event count in each spatial bin
        """
        # make sure region is specified with catalog
        if self.event_count == 0:
            return numpy.zeros(self.region.num_nodes)

        if self.region is None:
            raise CSEPSchedulerException("Cannot create binned rates without region information.")

        # todo: this should be routed through self.region to allow for different types of regions
        output = regions._bin_catalog_spatial_counts(self.get_longitudes(),
                                                     self.get_latitudes(),
                                                     self.region.num_nodes,
                                                     self.region.mask,
                                                     self.region.idx_map,
                                                     self.region.xs,
                                                     self.region.ys)
        return output

    def spatial_event_probability(self):
        # make sure region is specified with catalog
        if self.event_count == 0:
            return numpy.zeros(self.region.num_nodes)

        if self.region is None:
            raise CSEPSchedulerException("Cannot create binned probabilities without region information.")

        output = regions._bin_catalog_probability(self.get_longitudes(),
                                                  self.get_latitudes(),
                                                  len(self.region.polygons),
                                                  self.region.mask,
                                                  self.region.idx_map,
                                                  self.region.xs,
                                                  self.region.ys)
        return output

    def magnitude_counts(self, mag_bins=None, retbins=False):
        """ Computes the count of events within mag_bins


        Args:
            mag_bins: uses csep.utils.constants.CSEP_MW_BINS as default magnitude bins
            retbins (bool): if this is true, return the bins used

        Returns:
            numpy.ndarray: showing the counts of hte events in each magnitude bin
        """
        # todo: keep track of events that are ignored
        if mag_bins is None:
            try:
                # a forecast is a type of region, but region does not need a magnitude
                mag_bins = self.region.magnitudes
            except AttributeError:
                # use default magnitude bins from csep
                mag_bins = CSEP_MW_BINS
                self.region.magnitudes = mag_bins
                self.region.num_mag_bins = len(mag_bins)

        out = numpy.zeros(len(mag_bins))
        if self.event_count == 0:
            if retbins:
                return (mag_bins, out)
            else:
                return out
        idx = bin1d_vec(self.get_magnitudes(), mag_bins, right_continuous=True)
        numpy.add.at(out, idx, 1)
        if retbins:
            return (mag_bins, out)
        else:
            return out

    def spatial_magnitude_counts(self, mag_bins=None, ret_skipped=False):
        """ Return counts of events in space-magnitude region.

        We figure out the index of the polygons and create a map that relates the spatial coordinate in the
        Cartesian grid with with the polygon in region.

        Args:
            mag_bins: magnitude bins (optional). tries to use magnitue bins associated with region
            ret_skipped (bool): if true, will return list of (lon, lat, mw) tuple of skipped points


        Returns:
            output: unnormalized event count in each bin, 1d ndarray where index corresponds to midpoints

        """

        # make sure region is specified with catalog
        if self.region is None:
            raise CSEPCatalogException("Cannot create binned rates without region information.")

        # short-circuit if zero-events in catalog... return array of zeros
        if self.event_count == 0:
            n_poly = self.region.num_nodes
            n_mws = self.region.num_mag_bins
            output = numpy.zeros((n_poly, n_mws))
            skipped = []

        else:
            if mag_bins is None:
                try:
                    # a forecast is a type of region, but region does not need a magnitude
                    mag_bins = self.region.magnitudes
                except AttributeError:
                    # use default magnitude bins from csep
                    mag_bins = CSEP_MW_BINS
                    self.region.magnitudes = mag_bins
                    self.region.num_mag_bins = len(mag_bins)

            # compute if not
            # todo: this should be routed through self.region to allow for different types of regions
            output, skipped = regions._bin_catalog_spatio_magnitude_counts(self.get_longitudes(),
                                                                           self.get_latitudes(),
                                                                           self.get_magnitudes(),
                                                                           self.region.num_nodes,
                                                                           self.region.mask,
                                                                           self.region.idx_map,
                                                                           self.region.xs,
                                                                           self.region.ys,
                                                                           mag_bins)
        if ret_skipped:
            return output, skipped
        else:
            return output

    def length_in_seconds(self):
        """ Returns catalog length in seconds assuming that the catalog is sorted by time. """
        dts = self.get_datetimes()
        elapsed_time = (dts[-1] - dts[0]).total_seconds()
        return elapsed_time

    def get_bvalue(self, mag_bins=None, return_error=True):
        """
        Estimates the b-value of a catalog from Marzocchi and Sandri (2003). First, tries to use the magnitude bins
        provided to the function. If those are not provided, tries the magnitude bins associated with the region.
        If that fails, uses the default magnitude bins provided in constants.

        Args:
            reterr (bool): returns errors
            mag_bins (list or array_like): monotonically increasing set of magnitude bin edges

        Returns:
            bval (float): b-value
            err (float): std. err
        """

        if self.get_number_of_events() == 0:
            return None
        # this might fail if magnitudes are not aligned
        if mag_bins is None:
            try:
                mag_bins = self.region.magnitudes
            except AttributeError:
                mag_bins = CSEP_MW_BINS
        mws = discretize(self.get_magnitudes(), mag_bins)
        dmw = mag_bins[1] - mag_bins[0]

        # compute the p term from eq 3.10 in marzocchi and sandri [2003]
        def p():
            top = dmw
            bottom = numpy.mean(mws) - numpy.min(mws)
            # this might happen if all mags are the same, or 1 event in catalog
            if bottom == 0:
                return None
            return 1 + top / bottom

        bottom = numpy.log(10) * dmw
        p = p()
        if p is None:
            return None
        bval = 1.0 / bottom * numpy.log(p)
        if return_error:
            err = (1 - p) / (numpy.log(10) * dmw * numpy.sqrt(self.event_count * p))
            return (bval, err)
        else:
            return bval


class CSEPCatalog(AbstractBaseCatalog):
    """
    Standard catalog class for PyCSEP catalog operations.

    """

    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<i8'),
                         ('latitude', '<f8'),
                         ('longitude', '<f8'),
                         ('depth', '<f8'),
                         ('magnitude', '<f8')])

    def __init__(self, **kwargs):

        """ Standard catalog format for CSEP catalogs. Primary event data are stored in structured numpy array. Additional
            metadata are available by the event_id in the catalog metadata information.

        Args:
            filename: location of catalog
            catalog (numpy.ndarray or eventlist): catalog data
            catalog_id: catalog id number (used for stochastic event set forecasts)
            format: identification used for serialization
            name: human readable name of catalog
            region: spatial and magnitude region
            compute_stats: whether statistics should be computed for the catalog
            filters (str or list): filtering operations to apply to the catalog
            metadata (dict): additional information for events
            date_accessed (str): time string when catalog was accessed
        """
        super().__init__(**kwargs)

    @classmethod
    def load_ascii_catalogs(cls, filename, **kwargs):
        """ Loads multiple CSEP catalogs in ASCII format.

        This function can load multiple catalogs stored in a single file or directories. This typically called to
        load a catalog-based forecast.

        Args:
            filename (str): filepath or directory of catalog files
            **kwargs (dict): passed to class constructor

        Return:
            yields CSEPCatalog class
        """

        def parse_filename(filename):
            # this works for unix
            basename = str(os.path.basename(filename.rstrip('/')).split('.')[0])
            split_fname = basename.split('_')
            name = split_fname[0]
            start_time = strptime_to_utc_datetime(split_fname[1], format="%Y-%m-%dT%H-%M-%S-%f")
            return (name, start_time)

        def is_header_line(line):
            if line[0] == 'lon':
                return True
            else:
                return False

        name_from_file, start_time = parse_filename(filename)
        # overwrite filename, if user specifies
        kwargs.setdefault('name', name_from_file)
        # handle all catalogs in single file
        if os.path.isfile(filename):
            with open(filename, 'r', newline='') as input_file:
                catalog_reader = csv.reader(input_file, delimiter=',')
                # csv treats everything as a string convert to correct types
                events = []
                # all catalogs should start at zero
                prev_id = None
                for line in catalog_reader:
                    # skip header line on first read if included in file
                    if prev_id is None:
                        if is_header_line(line):
                            continue
                    # convert to correct types
                    lon = float(line[0])
                    lat = float(line[1])
                    magnitude = float(line[2])
                    # maybe fractional seconds are not included
                    try:
                        origin_time = strptime_to_utc_epoch(line[3], format='%Y-%m-%dT%H:%M:%S.%f')
                    except ValueError:
                        origin_time = strptime_to_utc_epoch(line[3], format='%Y-%m-%dT%H:%M:%S')
                    depth = float(line[4])
                    catalog_id = int(line[5])
                    event_id = line[6]
                    # first event is when prev_id is none, catalog_id should always start at zero
                    if prev_id is None:
                        prev_id = 0
                        # if the first catalog doesn't start at zero
                        if catalog_id != prev_id:
                            prev_id = catalog_id
                            # store this event for next time
                            events = [(event_id, origin_time, lat, lon, depth, magnitude)]
                            for id in range(catalog_id):
                                yield cls(data=[], catalog_id=id, **kwargs)
                    # deal with cases of events
                    if catalog_id == prev_id:
                        prev_id = catalog_id
                        events.append((event_id, origin_time, lat, lon, depth, magnitude))
                    # create and yield class if the events are from different catalogs
                    elif catalog_id == prev_id + 1:
                        catalog = cls(data=events, catalog_id=prev_id, **kwargs)
                        prev_id = catalog_id
                        # add first event to new event list
                        events = [(event_id, origin_time, lat, lon, depth, magnitude)]
                        yield catalog
                    # this implies there are empty catalogs, because they are not listed in the ascii file
                    elif catalog_id > prev_id + 1:
                        catalog = cls(data=events, catalog_id=prev_id, **kwargs)
                        # add event to new event list
                        events = [(event_id, origin_time, lat, lon, depth, magnitude)]
                        # if prev_id = 0 and catalog_id = 2, then we skipped one catalog. thus, we skip catalog_id - prev_id - 1 catalogs
                        num_empty_catalogs = catalog_id - prev_id - 1
                        # create empty catalog classes
                        for id in range(num_empty_catalogs):
                            yield cls(data=[], catalog_id=catalog_id - num_empty_catalogs + id, **kwargs)
                        # finally we want to yield the buffered catalog to preserve order
                        prev_id = catalog_id
                        yield catalog
                    else:
                        raise ValueError(
                            "catalog_id should be monotonically increasing and events should be ordered by catalog_id")
                # yield final catalog, note: since this is just loading catalogs, it has no idea how many should be there
                yield cls(data=events, catalog_id=prev_id, **kwargs)

        if os.path.isdir(filename):
            raise NotImplementedError("reading from directory or batched files not implemented yet!")

    @classmethod
    def load_catalog(cls, filename, loader=csep_ascii, **kwargs):
        """ Loads catalog stored in CSEP1 ascii format """
        catalog_id = None
        try:
            event_list, catalog_id = loader(filename, return_catalog_id=True)
        except TypeError:
            event_list = loader(filename)
        new_class = cls(data=event_list, catalog_id=catalog_id, **kwargs)
        return new_class


class UCERF3Catalog(AbstractBaseCatalog):
    """
    Catalog written from UCERF3-ETAS binary format

    :var header_dtype: numpy.dtype description of synthetic catalog header.
    :var event_dtype: numpy.dtype description of ucerf3 catalog format
    """
    # binary format of UCERF3 catalog
    header_dtype = numpy.dtype([("file_version", ">i2"), ("catalog_size", ">i4")])

    def __init__(self, **kwargs):
        # initialize parent constructor
        super().__init__(**kwargs)

    @classmethod
    def load_catalogs(cls, filename, **kwargs):
        """
        Loads catalogs based on the merged binary file format of UCERF3. File format is described at
        https://scec.usc.edu/scecpedia/CSEP2_Storing_Stochastic_Event_Sets#Introduction.

        There is also the load_catalog method that will work on the individual binary output of the UCERF3-ETAS
        model.

        Args:
            filename (str): filename of binary stochastic event set
            kwargs (dict): keyword arguments to pass to class constructor
        Returns:
            list of catalogs of type UCERF3Catalog
        """
        with open(filename, 'rb') as catalog_file:
            # parse 4byte header from merged file
            number_simulations_in_set = numpy.fromfile(catalog_file, dtype='>i4', count=1)[0]
            # load all catalogs from merged file
            for catalog_id in range(number_simulations_in_set):
                dtype = cls._get_header_dtype(version)
                version = numpy.fromfile(catalog_file, dtype=">i2", count=1)[0]
                header = numpy.fromfile(catalog_file, dtype=dtype, count=1)
                catalog_size = header['catalog_size'][0]
                # read catalog
                catalog = numpy.fromfile(catalog_file, dtype=cls._get_catalog_dtype(version), count=catalog_size)
                # add column that stores catalog_id in case we want to store in database
                u3_catalog = cls(filename=filename, data=catalog, catalog_id=catalog_id, **kwargs)
                u3_catalog.dtype = dtype
                yield u3_catalog

    @classmethod
    def load_catalog(cls, filename, loader=None, **kwargs):
        version = numpy.fromfile(filename, dtype=">i2", count=1)[0]
        header = numpy.fromfile(filename, dtype=cls._get_header_dtype(version), count=1)
        catalog_size = header['catalog_size'][0]
        # assign dtype to make sure that its bound to the instance of the class
        dtype = cls._get_catalog_dtype(version)
        event_list = numpy.fromfile(filename, dtype=cls._get_catalog_dtype(version), count=catalog_size)
        new_class = cls(filename=filename, data=event_list, **kwargs)
        new_class.dtype = dtype
        return new_class

    def get_csep_format(self):
        n = len(self.catalog)
        # allocate array for csep catalog
        csep_catalog = numpy.zeros(n, dtype=CSEPCatalog.dtype)
        for i, event in enumerate(self.catalog):
            csep_catalog[i] = (i,
                               event['origin_time'],
                               event['latitude'],
                               event['longitude'],
                               event['depth'],
                               event['magnitude'])
        return CSEPCatalog(data=csep_catalog, catalog_id=self.catalog_id, filename=self.filename, format='csep',
                           name=self.name, region=self.region, compute_stats=self.compute_stats, filters=self.filters,
                           metadata=self.metadata, date_accessed=self.date_accessed)

    @staticmethod
    def _get_catalog_dtype(version):
        """
        Get catalog dtype from version number

        Args:
            version:

        Returns:

        """

        if version == 1:
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

        elif version >= 2:
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
                                 ("grid_node_index", ">i4"),
                                 ("etas_k", ">f8")])

        else:
            raise ValueError("unknown catalog version, cannot read catalog.")

        return dtype

    @staticmethod
    def _get_header_dtype(version):

        if version == 1 or version == 2:
            dtype = numpy.dtype([("catalog_size", ">i4")])

        elif version >= 3:
            dtype = numpy.dtype([("num_orignal_ruptures", ">i4"),
                                 ("seed", ">i8"),
                                 ("index", ">i4"),
                                 ("hist_rupt_start_id", ">i4"),
                                 ("hist_rupt_end_id", ">i4"),
                                 ("trig_rupt_start_id", ">i4"),
                                 ("trig_rupt_end_id", ">i4"),
                                 ("sim_start_epoch", ">i8"),
                                 ("sim_end_epoch", ">i8"),
                                 ("num_spont", ">i4"),
                                 ("num_supraseis", ">i4"),
                                 ("min_mag", ">f8"),
                                 ("max_mag", ">f8"),
                                 ("catalog_size", ">i4")])
        else:
            raise ValueError("unknown catalog version, cannot parse catalog header.")
        return dtype

# helps to parse time-strings
def _none_or_datetime(value):
    if isinstance(value, datetime.datetime):
        return value
    if value is not None:
        format = parse_string_format(value)
        value = strptime_to_utc_datetime(value, format=format)
    return value
