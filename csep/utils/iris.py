# python imports
from datetime import datetime
from urllib import request
from urllib.parse import urlencode

# PyCSEP imports
from csep.utils.time_utils import datetime_to_utc_epoch

HOST_CATALOG = "https://service.iris.edu/fdsnws/event/1/query?"
TIMEOUT = 180


def gcmt_search(format='text',
                starttime=None,
                endtime=None,
                updatedafter=None,
                minlatitude=None,
                maxlatitude=None,
                minlongitude=None,
                maxlongitude=None,
                latitude=None,
                longitude=None,
                maxradius=None,
                catalog='GCMT',
                contributor=None,
                maxdepth=1000,
                maxmagnitude=10.0,
                mindepth=-100,
                minmagnitude=0,
                offset=1,
                orderby='time-asc',
                host=None,
                verbose=False):
    """Search the IRIS database for events matching input criteria.
    This search function is a wrapper around the ComCat Web API described here:
    https://service.iris.edu/fdsnws/event/1/

    This function returns a list of SummaryEvent objects, described elsewhere in this package.
    Args:
        starttime (datetime):
            Python datetime - Limit to events on or after the specified start time.
        endtime (datetime):
            Python datetime - Limit to events on or before the specified end time.
        updatedafter (datetime):
           Python datetime - Limit to events updated after the specified time.
        minlatitude (float):
            Limit to events with a latitude larger than the specified minimum.
        maxlatitude (float):
            Limit to events with a latitude smaller than the specified maximum.
        minlongitude (float):
            Limit to events with a longitude larger than the specified minimum.
        maxlongitude (float):
            Limit to events with a longitude smaller than the specified maximum.
        latitude (float):
            Specify the latitude to be used for a radius search.
        longitude (float):
            Specify the longitude to be used for a radius search.
        maxradius (float):
            Limit to events within the specified maximum number of degrees
            from the geographic point defined by the latitude and longitude parameters.
        catalog (str):
            Limit to events from a specified catalog.
        contributor (str):
            Limit to events contributed by a specified contributor.
        maxdepth (float):
            Limit to events with depth less than the specified maximum.
        maxmagnitude (float):
            Limit to events with a magnitude smaller than the specified maximum.
        mindepth (float):
            Limit to events with depth more than the specified minimum.
        minmagnitude (float):
            Limit to events with a magnitude larger than the specified minimum.
        offset (int):
            Return results starting at the event count specified, starting at 1.
        orderby (str):
            Order the results. The allowed values are:
            - time order by origin descending time
            - time-asc order by origin ascending time
            - magnitude order by descending magnitude
            - magnitude-asc order by ascending magnitude
        host (str):
            Replace default ComCat host (earthquake.usgs.gov) with a custom host.
    Returns:
        list: List of SummaryEvent() objects.
    """

    # getting the inputargs must be the first line of the method!
    inputargs = locals().copy()
    newargs = {}

    for key, value in inputargs.items():
        if value is True:
            newargs[key] = 'true'
            continue
        if value is False:
            newargs[key] = 'false'
            continue
        if value is None:
            continue
        newargs[key] = value

    del newargs['verbose']

    events = _search_gcmt(**newargs)

    return events


def _search_gcmt(**_newargs):
    """
    Performs de-query at ISC API and returns event list and access date

    """
    paramstr = urlencode(_newargs)
    url = HOST_CATALOG + paramstr
    fh = request.urlopen(url, timeout=TIMEOUT)
    data = fh.read().decode('utf8').split('\n')
    fh.close()
    eventlist = []
    for line in data[1:]:
        line_ = line.split('|')
        if len(line_) != 1:
            id_ = line_[0]
            time_ = datetime.fromisoformat(line_[1])
            dt = datetime_to_utc_epoch(time_)
            lat = float(line_[2])
            lon = float(line_[3])
            depth = float(line_[4])
            mag = float(line_[10])
            eventlist.append((id_, dt, lat, lon, depth, mag))

    return eventlist