
# python imports
from datetime import datetime, timedelta
from urllib import request
from urllib.parse import  urlencode
import json



class SummaryEvent_gns(object):
    """Wrapper around summary feature as returned by ComCat GeoJSON search results.
    """

    def __init__(self, feature):
        """Instantiate a SummaryEvent object with a feature.
        See summary documentation here:
        https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
        Args:
            feature (dict): GeoJSON feature as described at above URL.
        """
        self._jdict = feature.copy()
    @property
    def url(self):
        """ComCat URL.
        Returns:
            str: ComCat URL
        """
        url_template= "https://www.geonet.org.nz/earthquake/"
        return url_template + self._jdict['properties']['publicid']

    @property
    def latitude(self):
        """Authoritative origin latitude.
        Returns:
            float: Authoritative origin latitude.
        """
        return self._jdict['geometry']['coordinates'][1]

    @property
    def longitude(self):
        """Authoritative origin longitude.
        Returns:
            float: Authoritative origin longitude.
        """
        return self._jdict['geometry']['coordinates'][0]

    @property
    def depth(self):
        """Authoritative origin depth.
        Returns:
            float: Authoritative origin depth.
        """
        return self._jdict['properties']['depth']

    @property
    def id(self):
        """Authoritative origin ID.
        Returns:
            str: Authoritative origin ID.
        """
        ## comcat has an id key in each feature, whereas bsi has eventId within the properties dict
        try:
            return self._jdict['properties']['publicid']
        except:
            return self._jdict['properties']['eventId']

    @property
    def time(self):
        """Authoritative origin time.
        Returns:
            datetime: Authoritative origin time.
        """
        from obspy import UTCDateTime
        time_in_msec = self._jdict['properties']['origintime']
        # Comcat gives the event time in a ms timestamp, whereas bsi in datetime isoformat
        if isinstance(time_in_msec, str):
            event_dtime = UTCDateTime(time_in_msec)
            time_in_msec = event_dtime.timestamp * 1000
        time_in_sec = time_in_msec // 1000
        msec = time_in_msec - (time_in_sec * 1000)
        dtime = datetime.utcfromtimestamp(time_in_sec)
        dt = timedelta(milliseconds=msec)
        dtime = dtime + dt
        return dtime

    @property
    def magnitude(self):
        """Authoritative origin magnitude.
        Returns:
            float: Authoritative origin magnitude.
        """
        return self._jdict['properties']['magnitude']

    def __repr__(self):
        tpl = (self.id, str(self.time), self.latitude,
               self.longitude, self.depth, self.magnitude)
        return '%s %s (%.3f,%.3f) %.1f km M%.1f' % tpl


def gns_search(
    starttime=None, 
    endtime=None,
    minlatitude=-47, 
    maxlatitude=-34,
    minlongitude=164, 
    maxlongitude=181,
    minmagnitude=2.95,
    maxmagnitude=None,
    maxdepth=45.5,
    mindepth=None):

    """Search the Geonet database for events matching input criteria.
    This search function is a wrapper around the Geonet Web API described here:
    https://quakesearch.geonet.org.nz/

    Note:   
        Geonet has limited search parameters compered to ComCat search parameters, 
        hence the need for a new function 
    Args:
        starttime (datetime):
            Python datetime - Limit to events on or after the specified start time.
        endtime (datetime):
            Python datetime - Limit to events on or before the specified end time.
        minlatitude (float):
            Limit to events with a latitude larger than the specified minimum.
        maxlatitude (float):
            Limit to events with a latitude smaller than the specified maximum.
        minlongitude (float):
            Limit to events with a longitude larger than the specified minimum.
        maxlongitude (float):
            Limit to events with a longitude smaller than the specified maximum.
        maxdepth (float):
            Limit to events with depth less than the specified maximum.
        maxmagnitude (float):
            Limit to events with a magnitude smaller than the specified maximum.
        mindepth (float):
            Limit to events with depth more than the specified minimum.
        minmagnitude (float):
            Limit to events with a magnitude larger than the specified minimum.
        host (str):
            Replace default ComCat host (earthquake.usgs.gov) with a custom host.
    Returns:
        list: List of dictionary with event info.
    """
    # getting the inputargs must be the first line of the method!

    TIMEFMT = '%Y-%m-%dT%H:%M:%S'
    TIMEOUT = 120  # how long do we wait for a url to return?
    try:
        newargs = {}
        newargs["bbox"] = f'{minlongitude},{minlatitude},{maxlongitude},{maxlatitude}'
        newargs["minmag"] = f'{minmagnitude}'
        newargs["maxdepth"] = f'{maxdepth}'
        newargs["startdate"] = starttime.strftime(TIMEFMT) 
        newargs["enddate"] = endtime.strftime(TIMEFMT)
        if maxmagnitude is not None:
            newargs["maxmag"] = f'{maxmagnitude}'
        if mindepth is not None:
            newargs["mindepth"] = f'{mindepth}'

     
        paramstr = urlencode(newargs)
        template = "https://quakesearch.geonet.org.nz/geojson?"
        url = template + '&' + paramstr
        # print(url)
        try:
            fh = request.urlopen(url, timeout=TIMEOUT)
            data = fh.read().decode('utf8')
            fh.close()
            jdict = json.loads(data)
            events = []
            for feature in jdict['features']:
                events.append(SummaryEvent_gns(feature))
                # tpl = (feature['properties']['publicid'], 
                #         str(time_gns(feature['properties']['origintime'])), 
                #         feature['geometry']['coordinates'][1],
                #         feature['geometry']['coordinates'][0], 
                #         feature['properties']['depth'],
                #         feature['properties']['magnitude']
                #         # url_template + feature['properties']['publicid']
                #         )
                # events.append('%s %s (%.3f,%.3f) %.1f km M%.1f' % tpl)
                # edict = OrderedDict()
                # edict['id'] = feature['properties']['publicid']
                # edict['time'] = feature['properties']['origintime']
                # edict['location'] = "New Zealand"
                # edict['latitude'] = feature['geometry']['coordinates'][0]
                # edict['longitude'] = feature['geometry']['coordinates'][0]
                # edict['depth'] = feature['properties']['depth'] 
                # edict['magnitude'] = feature['properties']['magnitude'] 
                # edict['url'] = url_template + feature['properties']['publicid']
                # events.append(edict)
            # events.append(SummaryEvent_gns(feature))
        except Exception as msg:
            raise Exception(
                'Error downloading data from url %s.  "%s".' % (url, msg))
    
        return events
    except ValueError as e:
        if len(e.args) > 0 and  'Invalid isoformat string' in e.args[0]:
            print("Check the input date format. It should follow YYYY-MM-DD \
                    and is should not be empty")