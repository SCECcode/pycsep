import datetime
import math
import re
import warnings
import enum
import csv
from itertools import zip_longest
import os

# Third-party imports
import numpy

# PyCSEP imports
from csep.utils.time_utils import strptime_to_utc_datetime, strptime_to_utc_epoch, datetime_to_utc_epoch
from csep.utils.comcat import search
from csep.core.exceptions import CSEPIOException

def ndk(filename):
    """
    Reads an NDK file to a tuple of events.

    This code was modified from the obspy v1.2.2 implementation to work with CSEP Catalog objects. The original source
    code can be found at https://github.com/obspy/obspy/blob/master/obspy/io/ndk/core.py.

    Args:
        filename: file or file-like object
    """
    # this function first parses the data into a human readable dict with appropriate values and then finally returns a
    # CSEP catalog object.

    def _read_lines(line1, line2, line3, line4, line5):
        # First line: Hypocenter line
        # [1-4]   Hypocenter reference catalog (e.g., PDE for USGS location,
        #         ISC for #ISC catalog, SWE for surface-wave location,
        #         [Ekstrom, BSSA, 2006])
        # [6-15]  Date of reference event
        # [17-26] Time of reference event
        # [28-33] Latitude
        # [35-41] Longitude
        # [43-47] Depth
        # [49-55] Reported magnitudes, usually mb and MS
        # [57-80] Geographical location (24 characters)
        rec = {}
        rec["hypocenter_reference_catalog"] = line1[:4].strip()
        rec["date"] = line1[5:15].strip()
        rec["time"] = line1[16:26]
        rec["hypo_lat"] = float(line1[27:33])
        rec["hypo_lng"] = float(line1[34:41])
        rec["hypo_depth_in_km"] = float(line1[42:47])
        rec["mb"], rec["MS"] = map(float, line1[48:55].split())
        rec["location"] = line1[56:80].strip()

        # Second line: CMT info (1)
        # [1-16]  CMT event name. This string is a unique CMT-event identifier.
        #         Older events have 8-character names, current ones have
        #         14-character names.  See note (1) below for the naming
        #         conventions used.
        # [18-61] Data used in the CMT inversion. Three data types may be used:
        #         Long-period body waves (B), Intermediate-period surface waves
        #         (S), and long-period mantle waves (M). For each data type,
        #         three values are given: the number of stations used, the
        #         number  of components used, and the shortest period used.
        # [63-68] Type of source inverted for:
        #         "CMT: 0" - general moment tensor;
        #         "CMT: 1" - moment tensor with constraint of zero trace
        #             (standard);
        #         "CMT: 2" - double-couple source.
        # [70-80] Type and duration of moment-rate function assumed in the
        #         inversion.  "TRIHD" indicates a triangular moment-rate
        #         function, "BOXHD" indicates a boxcar moment-rate function.
        #         The value given is half the duration of the moment-rate
        #         function. This value is assumed in the inversion, following a
        #         standard scaling relationship (see note (2) below), and is
        #         not derived from the analysis.
        rec["cmt_event_name"] = line2[:16].strip()

        data_used = line2[17:61].strip()
        # Use regex to get the data used in case the data types are in a
        # different order.
        data_used = re.findall(r"[A-Z]:\s*\d+\s+\d+\s+\d+", data_used)
        rec["data_used"] = []
        for data in data_used:
            data_type, count = data.split(":")
            if data_type == "B":
                data_type = "body waves"
            elif data_type == "S":
                data_type = "surface waves"
            elif data_type == "M":
                data_type = "mantle waves"
            else:
                msg = "Unknown data type '%s'." % data_type
                raise ValueError(msg)

            sta, comp, period = count.strip().split()

            rec["data_used"].append({
                "wave_type": data_type,
                "station_count": int(sta),
                "component_count": int(comp),
                "shortest_period": float(period)
            })

        source_type = line2[62:68].strip().upper().replace(" ", "")
        if source_type == "CMT:0":
            rec["source_type"] = "general"
        elif source_type == "CMT:1":
            rec["source_type"] = "zero trace"
        elif source_type == "CMT:2":
            rec["source_type"] = "double couple"
        else:
            msg = "Unknown source type."
            raise ValueError(msg)

        mr_type, mr_duration = [i.strip() for i in line2[69:].split(":")]
        mr_type = mr_type.strip().upper()
        if mr_type == "TRIHD":
            rec["moment_rate_type"] = "triangle"
        elif mr_type == "BOXHD":
            rec["moment_rate_type"] = "box car"
        else:
            msg = "Moment rate function '%s' unknown." % mr_type
            raise ValueError(msg)

        # Specified as half the duration in the file.
        rec["moment_rate_duration"] = float(mr_duration) * 2.0

        # Third line: CMT info (2)
        # [1-58]  Centroid parameters determined in the inversion. Centroid
        #         time, given with respect to the reference time, centroid
        #         latitude, centroid longitude, and centroid depth. The value
        #         of each variable is followed by its estimated standard error.
        #         See note (3) below for cases in which the hypocentral
        #         coordinates are held fixed.
        # [60-63] Type of depth. "FREE" indicates that the depth was a result
        #         of the inversion; "FIX " that the depth was fixed and not
        #         inverted for; "BDY " that the depth was fixed based on
        #         modeling of broad-band P waveforms.
        # [65-80] Timestamp. This 16-character string identifies the type of
        #         analysis that led to the given CMT results and, for recent
        #         events, the date and time of the analysis. This is useful to
        #         distinguish Quick CMTs ("Q-"), calculated within hours of an
        #         event, from Standard CMTs ("S-"), which are calculated later.
        if line3[0:9] != "CENTROID:":
            raise IOError("parse error: file should have CENTROID ")
        numbers = [line3[10:18], line3[18:22], line3[22:29], line3[29:34],
                   line3[34:42], line3[42:47], line3[47:53], line3[53:58]]
        rec["centroid_time"], rec["centroid_time_error"], \
        rec["centroid_latitude"], rec["centroid_latitude_error"], \
        rec["centroid_longitude"], rec["centroid_longitude_error"], \
        rec["centroid_depth_in_km"], rec["centroid_depth_in_km_error"] = \
            map(float, numbers)
        type_of_depth = line3[59:63].strip().upper()

        if type_of_depth == "FREE":
            rec["type_of_centroid_depth"] = "from moment tensor inversion"
        elif type_of_depth == "FIX":
            rec["type_of_centroid_depth"] = "from location"
        elif type_of_depth == "BDY":
            rec["type_of_centroid_depth"] = "from modeling of broad-band P " \
                                            "waveforms"
        else:
            msg = "Unknown type of depth '%s'." % type_of_depth
            raise ValueError(msg)

        timestamp = line3[64:].strip().upper()
        rec["cmt_timestamp"] = timestamp
        if timestamp.startswith("Q-"):
            rec["cmt_type"] = "quick"
        elif timestamp.startswith("S-"):
            rec["cmt_type"] = "standard"
        # This is invalid but occurs a lot so we include it here.
        elif timestamp.startswith("O-"):
            rec["cmt_type"] = "unknown"
        else:
            msg = "Invalid CMT timestamp '%s' for event %s." % (
                timestamp, rec["cmt_event_name"])
            raise ValueError(msg)

        # Fourth line: CMT info (3)
        # [1-2]   The exponent for all following moment values. For example, if
        #         the exponent is given as 24, the moment values that follow,
        #         expressed in dyne-cm, should be multiplied by 10**24.
        # [3-80]  The six moment-tensor elements: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
        #         where r is up, t is south, and p is east. See Aki and
        #         Richards for conversions to other coordinate systems. The
        #         value of each moment-tensor element is followed by its
        #         estimated standard error. See note (4) below for cases in
        #         which some elements are constrained in the inversion.
        # Exponent converts to dyne*cm. To convert to N*m it has to be decreased
        # seven orders of magnitude.
        exponent = int(line4[:2]) - 7
        # Directly set the exponent instead of calculating it to enhance
        # precision.
        rec["m_rr"], rec["m_rr_error"], rec["m_tt"], rec["m_tt_error"], \
        rec["m_pp"], rec["m_pp_error"], rec["m_rt"], rec["m_rt_error"], \
        rec["m_rp"], rec["m_rp_error"], rec["m_tp"], rec["m_tp_error"] = \
            map(lambda x: float("%sE%i" % (x, exponent)), line4[2:].split())

        # Fifth line: CMT info (4)
        # [1-3]   Version code. This three-character string is used to track
        #         the version of the program that generates the "ndk" file.
        # [4-48]  Moment tensor expressed in its principal-axis system:
        #         eigenvalue, plunge, and azimuth of the three eigenvectors.
        #         The eigenvalue should be multiplied by 10**(exponent) as
        #         given on line four.
        # [50-56] Scalar moment, to be multiplied by 10**(exponent) as given on
        #         line four.
        # [58-80] Strike, dip, and rake for first nodal plane of the
        #         best-double-couple mechanism, repeated for the second nodal
        #         plane.  The angles are defined as in Aki and Richards. The
        #         format for this string should not be considered fixed.
        rec["version_code"] = line5[:3].strip()
        rec["scalar_moment"] = float(line5[49:56]) * (10 ** exponent)
        # Calculate the moment magnitude.
        rec["Mw"] = 2.0 / 3.0 * (math.log10(rec["scalar_moment"]) - 9.1)

        principal_axis = line5[3:48].split()
        rec["principal_axis"] = []
        for axis in zip(*[iter(principal_axis)] * 3):
            rec["principal_axis"].append({
                # Again set the exponent directly to avoid even more rounding
                # errors.
                "length": "%sE%i" % (axis[0], exponent),
                "plunge": float(axis[1]),
                "azimuth": float(axis[2])
            })

        nodal_planes = map(float, line5[57:].strip().split())
        rec["nodal_plane_1"] = {
            "strike": next(nodal_planes),
            "dip": next(nodal_planes),
            "rake": next(nodal_planes)
        }
        rec["nodal_plane_2"] = {
            "strike": next(nodal_planes),
            "dip": next(nodal_planes),
            "rake": next(nodal_planes)
        }

        return rec

    out = []

    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rt") as fh:
                data = fh.read()
        except Exception:
            try:
                data = filename.decode()
            except Exception:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode()

    # Create iterator that yields lines.
    def lines_iter():
        prev_line = -1
        while True:
            next_line = data.find("\n", prev_line + 1)
            if next_line < 0:
                break
            yield data[prev_line + 1: next_line]
            prev_line = next_line
        if len(data) > prev_line + 1:
            yield data[prev_line + 1:]

    # Loop over 5 lines at once.
    for _i, lines in enumerate(zip_longest(*[lines_iter()] * 5)):
        if None in lines:
            msg = "Skipped last %i lines. Not a multiple of 5 lines." % (
                lines.count(None))
            warnings.warn(msg, RuntimeWarning)
            continue

        # Parse the lines to a human readable dictionary.
        try:
            record = _read_lines(*lines)
        # need to handle the exception here.
        except (ValueError, IOError):
            # exc = traceback.format_exc()
            msg = (
                "Could not parse event %i (faulty file?). Will be "
                "skipped." % (_i + 1))
            warnings.warn(msg, RuntimeWarning)
            continue

        # Assemble the time for the reference origin.
        try:
            date_time_dict = _parse_datetime_to_zmap(record["date"], record["time"])
        except ValueError:
            msg = ("Invalid time in event %i. '%s' and '%s' cannot be "
                   "assembled to a valid time. Event will be skipped.") % \
                  (_i + 1, record["date"], record["time"])
            warnings.warn(msg, RuntimeWarning)
            continue

        # we are stripping off a significant amount of information from the gCMT catalog
        # if more information is required please use the obspy implementation
        dt = datetime.datetime(
            date_time_dict['year'],
            date_time_dict['month'],
            date_time_dict['day'],
            date_time_dict['hour'],
            date_time_dict['minute'],
            date_time_dict['second']
        )
        out_tup = (_i,
                   datetime_to_utc_epoch(dt),
                   record['hypo_lat'],
                   record['hypo_lng'],
                   record["hypo_depth_in_km"],
                   record["Mw"])
        out.append(out_tup)
    return out


def zmap_ascii(fname, delimiter=None):
    """
    Reads csep1 ascii format into numpy structured array. this can be passed into a catalog object constructor. Using

    $ catalog = csep.core.catalogs.CSEPCatalog(catalog=zmap_ascii(fname), **kwargs)

    Many of the catalogs from the CSEP1 testing center were empty indicating that no observed earthquakes were available
    during the time period of the catalog. In the case of an empty catalog, this function will return an empty numpy array. The
    catalog object should still be created, but it will contain zero events. Therefore it can still be used for evaluations
    and plotting as normal.

    The CSEP Format has the following dtype:

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

    Args:
        fname: absolute path to csep1 catalog file

    Returns:
        list: list of tuples representing above type, empty if no events were found
    """

    class ColumnIndex(enum.Enum):
        Longitude = 0
        Latitude = 1
        DecimalYear = 2
        Month = 3
        Day = 4
        Magnitude = 5
        Depth = 6
        Hour = 7
        Minute = 8
        Second = 9

        # Error columns
        HorizontalError = 10
        DepthError = 11
        MagnitudeError = 12

        NetworkName = 13
        NumColumns = 14

    # short-circuit for empty file
    if os.stat(fname).st_size == 0:
        return []

    # arrange file into list of tuples
    out = []
    zmap_catalog_data = numpy.loadtxt(fname, delimiter=delimiter)
    for event_id, line in enumerate(zmap_catalog_data):
        dt = datetime.datetime(
            line[ColumnIndex.DecimalYear],
            line[ColumnIndex.Month],
            line[ColumnIndex.Day],
            line[ColumnIndex.Hour],
            line[ColumnIndex.Minute],
            line[ColumnIndex.Second]
        )
        event_tuple = (
            event_id,
            datetime_to_utc_epoch(dt),
            line[ColumnIndex.Latitude],
            line[ColumnIndex.Longitude],
            line[ColumnIndex.Depth],
            line[ColumnIndex.Magnitude],
        )
        out.append(event_tuple)
    return out


def csep_ascii(fname, return_catalog_id=False):
    """ Reads single catalog in CSEP ascii format.

    Args:
        fname (str): filename of catalog
        return_catalog_id (bool): return the catalog id

    Returns:
        list of tuples containing event information or (eventlist, catalog_id)
    """

    def is_header_line(line):
        if line[0] == 'lon':
            return True
        else:
            return False

    def parse_datetime(dt_string):
        try:
            origin_time = strptime_to_utc_epoch(dt_string, format='%Y-%m-%dT%H:%M:%S.%f')
            return origin_time
        except:
            pass
        try:
            origin_time = strptime_to_utc_epoch(dt_string, format='%Y-%m-%dT%H:%M:%S')
            return origin_time
        except:
            pass
        raise CSEPIOException("Supported time-string formats are '%Y-%m-%dT%H:%M:%S.%f' and '%Y-%m-%dT%H:%M:%S'")

    with open(fname, 'r', newline='') as input_file:
        catalog_reader = csv.reader(input_file, delimiter=',')
        # csv treats everything as a string convert to correct types
        is_first_event = True
        events = []
        for i, line in enumerate(catalog_reader):
            # skip header line on first read if included in file
            if is_first_event and is_header_line(line):
                continue
            # convert to correct types
            lon = float(line[0])
            lat = float(line[1])
            magnitude = float(line[2])
            # maybe fractional seconds are not included
            origin_time = parse_datetime(line[3])
            depth = float(line[4])
            catalog_id = int(line[5])
            event_id = line[6]
            try:
                event_id = event_id.decode('utf-8')
            except:
                pass
            if not event_id:
                event_id = int(i)

            events.append((event_id, origin_time, lat, lon, depth, magnitude))
            is_first_event = False

        if not return_catalog_id:
            return events
        else:
            return events, catalog_id

def ingv_emrcmt(fname):
    """
    Reader for the INGV (Istituto Nazionale di Geofisica e Vulcanologia - Italy)  European-
    Mediterranean regional Centroid Moment Tensor Catalog.
    It reads a catalog in .csv format, directly downloaded from http://rcmt2.bo.ingv.it/ using the Catalog Search (Beta
    version).
    
    
    The CSEP Format has the following dtype:

    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<i8'),
                         ('latitude', '<f4'),
                         ('longitude', '<f4'),
                         ('depth', '<f4'),
                         ('magnitude', '<f4')])

    Dismiss events with typo errors in its magnitude, and repeated events with 
    same id.
    
    """

    ind = {'evcat_id': 0,
           'date': 1,
           'time': 2,
           'sec_dec': 3,
           'lat': 4,
           'lon': 5,
           'depth': 6,
           'Mw': 61}

    out = []
    evcat_id = []
    n_event = 0
    with open(fname) as file_:
        reader = csv.reader(file_)
        
        for n, line in enumerate(reader):
            try:
                date = line[ind['date']].replace('-', '/')
                time = line[ind['time']].replace(' ', '0')
                sec_frac = line[ind['sec_dec']].replace(' ', '')
                if time.endswith(':'):
                    time += '00'
                date_time_dict = _parse_datetime_to_zmap(date,
                                                         time + '.' + sec_frac)
            except ValueError:
                msg = ("Could not parse date/time string '%s' and '%s' to a valid "
                       "time" % (line[ind['date']], line[ind['time']]))
                warnings.warn(msg, RuntimeWarning)
                continue

            dt = datetime.datetime(
                date_time_dict['year'],
                date_time_dict['month'],
                date_time_dict['day'],
                date_time_dict['hour'],
                date_time_dict['minute'],
                date_time_dict['second']
            )
            if 0. < float(line[ind["Mw"]]) < 10.0:
                event_tuple = (
                    n_event,
                    datetime_to_utc_epoch(dt),
                    float(line[ind["lat"]]),
                    float(line[ind["lon"]]),
                    float(line[ind["depth"]]),
                    float(line[ind["Mw"]])
                )
                n_event += 1
                evcat_id.append(line[ind["evcat_id"]])
                out.append(event_tuple)
            else:
                pass
            
        rep_events = [i for i in range(len(evcat_id)) if i not in 
                          numpy.unique(numpy.array(evcat_id), 
                                       return_index=True)[1]]
        for rep_id in rep_events:
            out.pop(rep_id)
        print('Removed %i badly formatted events' % (n + 1 - n_event))
        print('Removed %i repeated events' % len(rep_events))
        
    return out


def ingv_horus(fname):
    """
    Reader for the INGV (Istituto Nazionale di Geofisica e Vulcanologia - Italy)
    Homogenized instrumental seismic catalog (HORUS)
    It reads a catalog in plain text format, directly downloaded from 
    http://horus.bo.ingv.it/. 

    
    The CSEP Format has the following dtype:

    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<i8'),
                         ('latitude', '<f4'),
                         ('longitude', '<f4'),
                         ('depth', '<f4'),
                         ('magnitude', '<f4')])

    """

    ind = {'year': (0,"<i4"),
           'month': (1,"<i4"),
           'day': (2,"<i4"),
           'hour': (3,"<i4"),
           'minute': (4,"<i4"),
           'second': (5,"<f4"),
           'lat': (6, "<f4"),
           'lon': (7,"<f4"),
           'depth': (8,"<f4"),
           'Mw': (9,"<f4")}
    out = []
    
    data = numpy.genfromtxt(fname, skip_header=1,
                            names=ind.keys(), 
                            usecols=[i[0] for i in ind.values()],
                            dtype=[i[1] for i in ind.values()])
    for n, line in enumerate(data):
        dt = datetime.timedelta(0,0,0)
        if line['second'] >= 60.:
            line['second'] -= 60.
            dt += datetime.timedelta(minutes=1)
        if line['minute'] >= 60.:
            dt += datetime.timedelta(hours=1)
            line['minute'] -= 60.
        if line['hour'] >= 24.:
            dt += datetime.timedelta(days=1)
            line['hour'] -= 24.
        Time = datetime.datetime(
                                line['year'],
                                line['month'],
                                line['day'],
                                line['hour'],
                                line['minute'],
                                line['second']
                               ) + dt
        event_tuple = (Time,
                       datetime_to_utc_epoch(Time),
                       float(line["lat"]),
                       float(line["lon"]),
                       float(line["depth"]),
                       float(line["Mw"])
                       )   
        out.append(event_tuple)

    return out

def jma_csv(fname):
    """ Read catalog stored in pre-processed JMA comma separated values format.

     Originally written by Thomas Beutin, but adapted by William Savran to use native Python libraries.
     """
    # template for timestamp format in JMA csv file:
    _timestamp_template = '%Y-%m-%dT%H:%M:%S.%f%z'
    # helper function to parse the timestamps:
    parse_date_string = lambda x: round(1000. * datetime.datetime.strptime(x, _timestamp_template).timestamp())
    # helper function to determine if line is a header
    is_header_line = lambda x: True if x[0] == 'timestamp' else False
    # parse csv into formatted eventlist
    is_first_event = True
    events = []
    with open(fname, 'r', newline='') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        # this format doesnt include an id so generate one as simple index
        for id, line in enumerate(csv_reader):
            if is_first_event and is_header_line(line):
                continue
            # parse line
            origin_time = parse_date_string(line[0])
            lon = float(line[1])
            lat = float(line[2])
            depth = float(line[3])
            magnitude = float(line[4])
            events.append((id, origin_time, lat, lon, depth, magnitude))
            is_first_event = False
    return events


def _query_comcat(start_time, end_time, min_magnitude=2.50,
                 min_latitude=31.50, max_latitude=43.00,
                 min_longitude=-125.40, max_longitude=-113.10, extra_comcat_params=None):
    """
    Return eventlist from ComCat web service.

    Args:
        start_time (datetime.datetime): start time of catalog query
        end_time (datetime.datetime): end time of catalog query
        min_magnitude (float): minimum magnitude of query
        min_latitude (float): minimum latitude of query
        max_latitude (float): maximum latitude of query
        min_longitude (float): minimum longitude of query
        max_longitude (float): maximum longitude of query
        extra_comcat_params (dict): additional parameters to pass to comcat search function

    Returns:
        eventlist
    """
    extra_comcat_params = extra_comcat_params or {}

    # get eventlist from Comcat
    eventlist = search(minmagnitude=min_magnitude,
        minlatitude=min_latitude, maxlatitude=max_latitude,
        minlongitude=min_longitude, maxlongitude=max_longitude,
        starttime=start_time, endtime=end_time, **extra_comcat_params)

    return eventlist

def _parse_datetime_to_zmap(date, time):
        """ Helping function to return datetime in zmap format.

        Args:
            date: string record from .ndk file
            time: string record from .ndk file

        Returns:
            out: dictionary following
                out_dict = {'year': year, 'month': month, 'day': day',
                            'hour': hour, 'minute': minute, 'second': second}
        """

        add_minute = False
        if ":60.0" in time:
            time = time.replace(":60.0", ":0.0")
            add_minute = True
        try:
            dt = strptime_to_utc_datetime(date + " " + time, format="%Y/%m/%d %H:%M:%S.%f")
        except (TypeError, ValueError):
            msg = ("Could not parse date/time string '%s' and '%s' to a valid "
                   "time" % (date, time))
            raise RuntimeError(msg)

        if add_minute:
            dt += datetime.timedelta(minutes=1)

        out = {}
        out['year'] = dt.year
        out['month'] = dt.month
        out['day'] = dt.day
        out['hour'] = dt.hour
        out['minute'] = dt.minute
        out['second'] = dt.second
        return out
