# python imports
from datetime import datetime, timedelta, timezone
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlencode
import ssl
import json
import time
from collections import OrderedDict
import re
from enum import Enum
import sys

# 3rd-party imports
import numpy as np
import pandas as pd

# note: should consider how to remove these dependencies; should we just delete the functionality
import dateutil
from obspy.core.event import read_events

# PyCSEP imports
from csep.utils.time_utils import HistoricTime

# url template for counting events
HOST = 'earthquake.usgs.gov'
SEARCH_TEMPLATE = 'https://[HOST]/fdsnws/event/1/query?format=geojson'
TIMEOUT = 120  # how long do we wait for a url to return?
TIMEFMT = '%Y-%m-%dT%H:%M:%S'
WAITSECS = 3  # number of seconds to wait after failing download before trying again
SEARCH_LIMIT = 20000  # maximum number of events ComCat will return in one search
URL_TEMPLATE = ('https://earthquake.usgs.gov/earthquakes/feed'
                '/v1.0/detail/[EVENTID].geojson')
# the search template for a detail event that may
# include one or both of includesuperseded/includedeleted.
SEARCH_DETAIL_TEMPLATE = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                          '?format=geojson&eventid=%s&'
                          'includesuperseded=%s&includedeleted=%s')


class VersionOption(Enum):
    LAST = 1
    FIRST = 2
    ALL = 3
    PREFERRED = 4

def search(starttime=None,
           endtime=None,
           updatedafter=None,
           minlatitude=None,
           maxlatitude=None,
           minlongitude=None,
           maxlongitude=None,
           latitude=None,
           longitude=None,
           maxradiuskm=None,
           maxradius=None,
           catalog=None,
           contributor=None,
           limit=20000,
           maxdepth=1000,
           maxmagnitude=10.0,
           mindepth=-100,
           minmagnitude=0,
           offset=1,
           orderby='time-asc',
           alertlevel=None,
           eventtype='earthquake',
           maxcdi=None,
           maxgap=None,
           maxmmi=None,
           maxsig=None,
           mincdi=None,
           minfelt=None,
           mingap=None,
           minsig=None,
           producttype=None,
           productcode=None,
           reviewstatus=None,
           host=None,
           enable_limit=False,
           verbose=False):
    """

    Search the ComCat database for events matching input criteria.
    This search function is a wrapper around the ComCat Web API described here:
    https://earthquake.usgs.gov/fdsnws/event/1/
    Some of the search parameters described there are NOT implemented here, usually because they do not
    apply to GeoJSON search results, which we are getting here and parsing into Python data structures.
    This function returns a list of SummaryEvent objects, described elsewhere in this package.

    Args:
        starttime (datetime): Limit to events on or after the specified start time.
        endtime (datetime): Limit to events on or before the specified end time.
        updatedafter (datetime): Limit to events updated after the specified time.
        minlatitude (float): Limit to events with a latitude larger than the specified minimum.
        maxlatitude (float): Limit to events with a latitude smaller than the specified maximum.
        minlongitude (float): Limit to events with a longitude larger than the specified minimum.
        maxlongitude (float): Limit to events with a longitude smaller than the specified maximum.
        latitude (float): Specify the latitude to be used for a radius search.
        longitude (float): Specify the longitude to be used for a radius search.
        maxradiuskm (float): Limit to events within the specified maximum number of kilometers
            from the geographic point defined by the latitude and longitude parameters.
        maxradius (float): Limit to events within the specified maximum number of degrees
            from the geographic point defined by the latitude and longitude parameters.
        catalog (str): Limit to events from a specified catalog.
        contributor (str): Limit to events contributed by a specified contributor.
        limit (int): Limit the results to the specified number of events.
            NOTE: this will be throttled by this Python API to the supported Web API limit of 20,000.
        maxdepth (float): Limit to events with depth less than the specified maximum.
        maxmagnitude (float): Limit to events with a magnitude smaller than the specified maximum.
        mindepth (float): Limit to events with depth more than the specified minimum.
        minmagnitude (float): Limit to events with a magnitude larger than the specified minimum.
        offset (int): Return results starting at the event count specified, starting at 1.
        orderby (str): Order the results. The allowed values are:

            - `time`: order by origin descending time
            - `time-asc`: order by origin ascending time
            - `magnitude`: order by descending magnitude
            - `magnitude-asc`: order by ascending magnitude

        alertlevel (str): Limit to events with a specific PAGER alert level. The allowed values are:

            - `green`: Limit to events with PAGER alert level "green".
            - `yellow`: Limit to events with PAGER alert level "yellow".
            - `orange`: Limit to events with PAGER alert level "orange".
            - `red`: Limit to events with PAGER alert level "red".

        eventtype (str): Limit to events of a specific type. NOTE: "earthquake" will filter non-earthquake events.
        maxcdi (float): Maximum value for Maximum Community Determined Intensity reported by DYFI.
        maxgap (float): Limit to events with no more than this azimuthal gap.
        maxmmi (float): Maximum value for Maximum Modified Mercalli Intensity reported by ShakeMap.
        maxsig (float): Limit to events with no more than this significance.
        mincdi (float): Minimum value for Maximum Community Determined Intensity reported by DYFI.
        minfelt (int): Limit to events with this many DYFI responses.
        mingap (float): Limit to events with no less than this azimuthal gap.
        minsig (float): Limit to events with no less than this significance.
        producttype (str): Limit to events that have this type of product associated. Example product types:

            - `moment-tensor`
            - `focal-mechanism`
            - `shakemap`
            - `losspager`
            - `dyfi`

        productcode (str): Return the event that is associated with the product code.
            The event will be returned even if the product code is not
            the preferred code for the event. Example product codes:

            - `nn00458749`
            - `at00ndf1fr`

        reviewstatus (str): Limit to events with a specific review status. The different review statuses are:

            - `automatic`: Limit to events with review status "automatic".
            - `reviewed`: Limit to events with review status "reviewed".

        host (str): Replace default ComCat host (earthquake.usgs.gov) with a custom host.
        enable_limit (bool): Enable 20,000 event search limit. Will turn off searching
            in segments, which is meant to safely avoid that limit.
            Use only when you are certain your search will be small.

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
    if newargs['limit'] > 20000:
        newargs['limit'] = 20000

    # remove the verbose element from the arguments
    del newargs['verbose']
    del newargs['enable_limit']
    if enable_limit:
        events = _search(**newargs)
        return events
    segments = _get_time_segments(starttime, endtime, newargs['minmagnitude'])
    events = []
    iseg = 1
    for stime, etime in segments:
        newargs['starttime'] = stime
        newargs['endtime'] = etime
        if verbose:
            sys.stderr.write(
                'Searching time segment %i: %s to %s\n' % (iseg, stime, etime))
        iseg += 1
        events += _search(**newargs)

    return events

def _get_time_segments(starttime, endtime, minmag):
    if starttime is None:
        starttime = HistoricTime.utcnow() - timedelta(days=30)
    if endtime is None:
        endtime = HistoricTime.utcnow()
    # earthquake frequency table: minmag:earthquakes per day
    freq_table = {0: 3000 / 7,
                  1: 3500 / 14,
                  2: 3000 / 18,
                  3: 4000 / 59,
                  4: 9000 / 151,
                  5: 3000 / 365,
                  6: 210 / 365,
                  7: 20 / 365,
                  8: 5 / 365,
                  9: 0.05 / 365}

    floormag = int(np.floor(minmag))
    ndays = (endtime - starttime).days + 1
    freq = freq_table[floormag]
    nsegments = int(np.ceil((freq * ndays) / SEARCH_LIMIT))
    days_per_segment = int(np.ceil(ndays / nsegments))
    segments = []
    startseg = starttime
    endseg = starttime
    while startseg <= endtime:
        endseg = min(endtime, startseg + timedelta(days_per_segment))
        segments.append((startseg, endseg))
        startseg += timedelta(days=days_per_segment, microseconds=1)
    return segments

def _search(**newargs):
    if 'starttime' in newargs:
        newargs['starttime'] = newargs['starttime'].strftime(TIMEFMT)
    if 'endtime' in newargs:
        newargs['endtime'] = newargs['endtime'].strftime(TIMEFMT)
    if 'updatedafter' in newargs:
        newargs['updatedafter'] = newargs['updatedafter'].strftime(TIMEFMT)
    if 'host' in newargs and newargs['host'] is not None:
        template = SEARCH_TEMPLATE.replace('[HOST]', newargs['host'])
        del newargs['host']
    else:
        template = SEARCH_TEMPLATE.replace('[HOST]', HOST)

    paramstr = urlencode(newargs)
    url = template + '&' + paramstr
    events = []
    # handle the case when they're asking for an event id
    if 'eventid' in newargs:
        return DetailEvent(url)

    try:
        fh = request.urlopen(url, timeout=TIMEOUT)
        data = fh.read().decode('utf8')
        fh.close()
        jdict = json.loads(data)
        events = []
        for feature in jdict['features']:
            events.append(SummaryEvent(feature))
    except HTTPError as htpe:
        if htpe.code == 503:
            try:
                time.sleep(WAITSECS)
                fh = request.urlopen(url, timeout=TIMEOUT)
                data = fh.read().decode('utf8')
                fh.close()
                jdict = json.loads(data)
                events = []
                for feature in jdict['features']:
                    events.append(SummaryEvent(feature))
            except Exception as msg:
                raise Exception(
                    'Error downloading data from url %s.  "%s".' % (url, msg))

    except ssl.SSLCertVerificationError as SSLe:
        # Fails to verify SSL certificate, when there is a hostname mismatch
        if SSLe.verify_code == 62:
            try:
                context = ssl._create_unverified_context()
                fh = request.urlopen(url, timeout=TIMEOUT, context=context)
                data = fh.read().decode('utf8')
                fh.close()
                jdict = json.loads(data)
                events = []
                for feature in jdict['features']:
                    events.append(SummaryEvent(feature))
            except Exception as msg:
                raise Exception(
                    'Error downloading data from url %s.  "%s".' % (url, msg))

    except URLError as URLe:
        # Fails to verify SSL certificate, when there is a hostname mismatch
        if (isinstance(URLe.reason, ssl.SSLCertVerificationError) and URLe.reason.verify_code == 62) \
                or (isinstance(URLe.reason, ssl.SSLError) and URLe.reason.errno == 5):
            try:
                context = ssl._create_unverified_context()
                fh = request.urlopen(url, timeout=TIMEOUT, context=context)
                data = fh.read().decode('utf8')
                fh.close()
                jdict = json.loads(data)
                events = []
                for feature in jdict['features']:
                    events.append(SummaryEvent(feature))
            except Exception as msg:
                raise Exception(
                    'Error downloading data from url %s.  "%s".' % (url, msg))

    except Exception as msg:
        raise Exception(
            'Error downloading data from url %s.  "%s".' % (url, msg))

    return events

class SummaryEvent(object):
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
    def location(self):
        """Earthquake location string.
        Returns:
            str: Earthquake location.
        """
        return self._jdict['properties']['place']

    @property
    def url(self):
        """ComCat URL.
        Returns:
            str: ComCat URL
        """
        return self._jdict['properties']['url']

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
        return self._jdict['geometry']['coordinates'][2]

    @property
    def id(self):
        """Authoritative origin ID.
        Returns:
            str: Authoritative origin ID.
        """
        ## comcat has an id key in each feature, whereas bsi has eventId within the properties dict
        try:
            return self._jdict['id']
        except:
            return self._jdict['properties']['eventId']

    @property
    def time(self):
        """Authoritative origin time.
        Returns:
            datetime: Authoritative origin time.
        """
        time_in_msec = self._jdict['properties']['time']
        # Comcat gives the event time in a ms timestamp, whereas bsi in datetime isoformat
        if isinstance(time_in_msec, str):
            event_dtime = datetime.fromisoformat(time_in_msec).replace(tzinfo=timezone.utc)
            time_in_msec = event_dtime.timestamp() * 1000
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
        return self._jdict['properties']['mag']

    def __repr__(self):
        tpl = (self.id, str(self.time), self.latitude,
               self.longitude, self.depth, self.magnitude)
        return '%s %s (%.3f,%.3f) %.1f km M%.1f' % tpl

    @property
    def properties(self):
        """List of summary event properties.
        Returns:
            list: List of summary event properties (retrievable
                  from object with [] operator).
        """
        return list(self._jdict['properties'].keys())

    def hasProduct(self, product):
        """Test to see whether a given product exists for this event.
        Args:
            product (str): Product to search for.
        Returns:
            bool: Indicates whether that product exists or not.
        """
        if product not in self._jdict['properties']['types'].split(',')[1:]:
            return False
        return True

    def hasProperty(self, key):
        """Test to see if property is present in list of properties.
        Args:
            key (str): Property to search for.
        Returns:
          bool: Indicates whether that key exists or not.
        """
        if key not in self._jdict['properties']:
            return False
        return True

    def __getitem__(self, key):
        """Extract SummaryEvent property using the [] operator.
        Args:
            key (str): Property to extract.
        Returns:
            str: Desired property.
        """
        if key not in self._jdict['properties']:
            raise AttributeError(
                'No property %s found for event %s.' % (key, self.id))
        return self._jdict['properties'][key]

    def getDetailURL(self):
        """Instantiate a DetailEvent object from the URL found in the summary.
        Returns:
            str: URL for detailed version of event.
        """
        durl = self._jdict['properties']['detail']
        return durl

    def getDetailEvent(self, includedeleted=False, includesuperseded=False):
        """Instantiate a DetailEvent object from the URL found in the summary.
        Args:
            includedeleted (bool): Boolean indicating wheather to return
                versions of products that have
                been deleted. Cannot be used with
                includesuperseded.
            includesuperseded (bool):
                Boolean indicating wheather to return versions of products
                that have been replaced by newer versions.
                Cannot be used with includedeleted.
        Returns:
            DetailEvent: Detailed version of SummaryEvent.
        """
        if includesuperseded and includedeleted:
            msg = ('includedeleted and includesuperseded '
                   'cannot be used together.')
            raise RuntimeError(msg)
        if not includedeleted and not includesuperseded:
            durl = self._jdict['properties']['detail']
            return DetailEvent(durl)
        else:
            true_false = {True: 'true', False: 'false'}
            deleted = true_false[includedeleted]
            superseded = true_false[includesuperseded]
            url = SEARCH_DETAIL_TEMPLATE % (self.id, superseded, deleted)
            return DetailEvent(url)

    def toDict(self):
        """Render the SummaryEvent origin information as an OrderedDict().
        Returns:
            dict: Containing fields:
               - id (string) Authoritative ComCat event ID.
               - time (datetime) Authoritative event origin time.
               - latitude (float) Authoritative event latitude.
               - longitude (float) Authoritative event longitude.
               - depth (float) Authoritative event depth.
               - magnitude (float) Authoritative event magnitude.
        """
        edict = OrderedDict()
        edict['id'] = self.id
        edict['time'] = self.time
        edict['location'] = self.location
        edict['latitude'] = self.latitude
        edict['longitude'] = self.longitude
        edict['depth'] = self.depth
        edict['magnitude'] = self.magnitude
        edict['url'] = self.url
        return edict

class DetailEvent(object):
    """Wrapper around detailed event as returned by ComCat GeoJSON search results.
    """

    def __init__(self, url):
        """Instantiate a DetailEvent object with a url pointing to detailed GeoJSON.
        See detailed documentation here:
        https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson_detail.php
        Args:
            url (str): String indicating a URL pointing to a detailed GeoJSON event.
        """
        try:
            fh = request.urlopen(url, timeout=TIMEOUT)
            data = fh.read().decode('utf-8')
            fh.close()
            self._jdict = json.loads(data)
        except HTTPError:
            try:
                fh = request.urlopen(url, timeout=TIMEOUT)
                data = fh.read().decode('utf-8')
                fh.close()
                self._jdict = json.loads(data)
            except Exception as msg:
                raise Exception('Could not connect to ComCat server - %s.' %
                                url).with_traceback(msg.__traceback__)

    def __repr__(self):
        tpl = (self.id, str(self.time), self.latitude,
               self.longitude, self.depth, self.magnitude)
        return '%s %s (%.3f,%.3f) %.1f km M%.1f' % tpl

    @property
    def location(self):
        """Earthquake location string.
        Returns:
            str: Earthquake location.
        """
        return self._jdict['properties']['place']

    @property
    def url(self):
        """ComCat URL.
        Returns:
            str: Earthquake URL.
        """
        return self._jdict['properties']['url']

    @property
    def detail_url(self):
        """ComCat Detailed URL (with JSON).
        Returns:
            str: Earthquake Detailed URL with JSON.
        """
        url = URL_TEMPLATE.replace('[EVENTID]', self.id)
        return url

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
        """
        return self._jdict['geometry']['coordinates'][2]

    @property
    def id(self):
        """Authoritative origin ID.
        Returns:
            str: Authoritative origin ID.
        """
        return self._jdict['id']

    @property
    def time(self):
        """Authoritative origin time.
        Returns:
            datetime: Authoritative origin time.
        """
        time_in_msec = self._jdict['properties']['time']
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
        return self._jdict['properties']['mag']

    @property
    def magtype(self):
        return self._jdict['properties']['magType']

    @property
    def properties(self):
        """List of detail event properties.
        Returns:
            list: List of summary event properties (retrievable from object with [] operator).
        """
        return list(self._jdict['properties'].keys())

    @property
    def products(self):
        """List of detail event properties.
        Returns:
            list: List of detail event products (retrievable from object with
                getProducts() method).
        """
        return list(self._jdict['properties']['products'].keys())

    def hasProduct(self, product):
        """Return a boolean indicating whether given product can be extracted from DetailEvent.
        Args:
            product (str): Product to search for.
        Returns:
            bool: Indicates whether that product exists or not.
        """
        if product in self._jdict['properties']['products']:
            return True
        return False

    def hasProperty(self, key):
        """Test to see whether a property with a given key is present in list of properties.
        Args:
            key (str): Property to search for.
        Returns:
            bool: Indicates whether that key exists or not.
        """
        if key not in self._jdict['properties']:
            return False
        return True

    def __getitem__(self, key):
        """Extract DetailEvent property using the [] operator.
        Args:
            key (str): Property to extract.
        Returns:
            str: Desired property.
        """
        if key not in self._jdict['properties']:
            raise AttributeError(
                'No property %s found for event %s.' % (key, self.id))
        return self._jdict['properties'][key]

    def toDict(self, catalog=None,
               get_tensors='preferred',
               get_moment_supplement=False,
               get_all_magnitudes=False,
               get_focals='preferred'):
        """Return origin, focal mechanism, and tensor information for a DetailEvent.
        Args:
            catalog (str): Retrieve the primary event information (time,lat,lon...) from the
                catalog given. If no source for this information exists, an
                AttributeError will be raised.
            get_tensors (str): Option of 'none', 'preferred', or 'all'.
            get_moment_supplement (bool): Boolean indicating whether derived origin and
                double-couple/source time information should be extracted
                (when available.)
            get_focals (str): String option of 'none', 'preferred', or 'all'.
        Returns:
            dict: OrderedDict with the same fields as returned by
                SummaryEvent.toDict(), *preferred* moment tensor and focal
                mechanism data.  If all magnitudes are requested, then
                those will be returned as well. Generally speaking, the
                number and name of the fields will vary by what data is available.
        """
        edict = OrderedDict()

        if catalog is None:
            edict['id'] = self.id
            edict['time'] = self.time
            edict['location'] = self.location
            edict['latitude'] = self.latitude
            edict['longitude'] = self.longitude
            edict['depth'] = self.depth
            edict['magnitude'] = self.magnitude
            edict['magtype'] = self._jdict['properties']['magType']
            edict['url'] = self.url
        else:
            try:
                phase_sources = []
                origin_sources = []
                if self.hasProduct('phase-data'):
                    phase_sources = [p.source for p in self.getProducts(
                        'phase-data', source='all')]
                if self.hasProduct('origin'):
                    origin_sources = [
                        o.source for o in self.getProducts('origin',
                                                           source='all')]
                if catalog in phase_sources:
                    phasedata = self.getProducts(
                        'phase-data', source=catalog)[0]
                elif catalog in origin_sources:
                    phasedata = self.getProducts('origin', source=catalog)[0]
                else:
                    msg = ('DetailEvent %s has no phase-data or origin '
                           'products for source %s')
                    raise AttributeError(msg % (self.id, catalog))
                edict['id'] = phasedata['eventsource'] + \
                    phasedata['eventsourcecode']
                edict['time'] = dateutil.parser.parse(phasedata['eventtime'])
                edict['location'] = self.location
                edict['latitude'] = float(phasedata['latitude'])
                edict['longitude'] = float(phasedata['longitude'])
                edict['depth'] = float(phasedata['depth'])
                edict['magnitude'] = float(phasedata['magnitude'])
                edict['magtype'] = phasedata['magnitude-type']
            except AttributeError as ae:
                raise ae

        if get_tensors == 'all':
            if self.hasProduct('moment-tensor'):
                tensors = self.getProducts(
                    'moment-tensor', source='all', version=VersionOption.ALL)
                for tensor in tensors:
                    supp = get_moment_supplement
                    tdict = _get_moment_tensor_info(tensor,
                                                    get_angles=True,
                                                    get_moment_supplement=supp)
                    edict.update(tdict)

        if get_tensors == 'preferred':
            if self.hasProduct('moment-tensor'):
                tensor = self.getProducts('moment-tensor')[0]
                supp = get_moment_supplement
                tdict = _get_moment_tensor_info(tensor, get_angles=True,
                                                get_moment_supplement=supp)
                edict.update(tdict)

        if get_focals == 'all':
            if self.hasProduct('focal-mechanism'):
                focals = self.getProducts(
                    'focal-mechanism', source='all', version=VersionOption.ALL)
                for focal in focals:
                    edict.update(_get_focal_mechanism_info(focal))

        if get_focals == 'preferred':
            if self.hasProduct('focal-mechanism'):
                focal = self.getProducts('focal-mechanism')[0]
                edict.update(_get_focal_mechanism_info(focal))

        # dependency on obspy for this function that we might not use
        if get_all_magnitudes:
            phase_data = self.getProducts('phase-data')[0]
            phase_url = phase_data.getContentURL('quakeml.xml')
            catalog = read_events(phase_url)
            event = catalog.events[0]
            imag = 1
            for magnitude in event.magnitudes:
                edict['magnitude%i' % imag] = magnitude.mag
                edict['magtype%i' %
                      imag] = magnitude.magnitude_type
                imag += 1

        return edict

    def getNumVersions(self, product_name):
        """Count versions of a product (origin, shakemap, etc.) available.
        Args:
            product_name (str): Name of product to query.
        Returns:
            int: Number of versions of a given product.
        """
        if not self.hasProduct(product_name):
            raise AttributeError(
                'Event %s has no product of type %s' % (self.id, product_name))
        return len(self._jdict['properties']['products'][product_name])

    def getProducts(self, product_name, source='preferred',
                    version=VersionOption.PREFERRED):
        """Retrieve a Product object from this DetailEvent.
        Args:
            product_name (str): Name of product (origin, shakemap, etc.) to retrieve.
            version (enum): A value from VersionOption (PREFERRED,FIRST,ALL).
            source (str): Any one of:
                - 'preferred' Get version(s) of products from preferred source.
                - 'all' Get version(s) of products from all sources.
                - Any valid source network for this type of product
                  ('us','ak',etc.)
        Returns:
          list: List of Product objects.
        """
        if not self.hasProduct(product_name):
            raise AttributeError(
                'Event %s has no product of type %s' % (self.id, product_name))

        products = self._jdict['properties']['products'][product_name]
        weights = [product['preferredWeight'] for product in products]
        sources = [product['source'] for product in products]
        times = [product['updateTime'] for product in products]
        indices = list(range(0, len(times)))
        df = pd.DataFrame(
            {'weight': weights, 'source': sources,
             'time': times, 'index': indices})
        # we need to add a version number column here, ordinal
        # sorted by update time, starting at 1
        # for each unique source.
        # first sort the dataframe by source and then time
        df = df.sort_values(['source', 'time'])
        df['version'] = 0
        psources = []
        pversion = 1
        for idx, row in df.iterrows():
            if row['source'] not in psources:
                psources.append(row['source'])
                pversion = 1
            df.loc[idx, 'version'] = pversion
            pversion += 1

        if source == 'preferred':
            idx = weights.index(max(weights))
            tproduct = self._jdict['properties']['products'][product_name][idx]
            prefsource = tproduct['source']
            df = df[df['source'] == prefsource]
            df = df.sort_values('time')
        elif source == 'all':
            df = df.sort_values(['source', 'time'])
        else:
            df = df[df['source'] == source]
            df = df.sort_values('time')

        # if we don't have any versions of products, raise an exception
        if not len(df):
            raise AttributeError('No products found for source "%s".' % source)

        products = []
        usources = set(sources)
        tproducts = self._jdict['properties']['products'][product_name]
        if source == 'all':  # dataframe includes all sources
            for source in usources:
                df_source = df[df['source'] == source]
                df_source = df_source.sort_values('time')
                if version == VersionOption.PREFERRED:
                    df_source = df_source.sort_values(['weight', 'time'])
                    idx = df_source.iloc[-1]['index']
                    pversion = df_source.iloc[-1]['version']
                    product = Product(product_name, pversion, tproducts[idx])
                    products.append(product)
                elif version == VersionOption.LAST:
                    idx = df_source.iloc[-1]['index']
                    pversion = df_source.iloc[-1]['version']
                    product = Product(product_name, pversion, tproducts[idx])
                    products.append(product)
                elif version == VersionOption.FIRST:
                    idx = df_source.iloc[0]['index']
                    pversion = df_source.iloc[0]['version']
                    product = Product(product_name, pversion, tproducts[idx])
                    products.append(product)
                elif version == VersionOption.ALL:
                    for idx, row in df_source.iterrows():
                        idx = row['index']
                        pversion = row['version']
                        product = Product(
                            product_name, pversion, tproducts[idx])
                        products.append(product)
                else:
                    raise(AttributeError(
                        'No VersionOption defined for %s' % version))
        else:  # dataframe only includes one source
            if version == VersionOption.PREFERRED:
                df = df.sort_values(['weight', 'time'])
                idx = df.iloc[-1]['index']
                pversion = df.iloc[-1]['version']
                product = Product(
                    product_name, pversion, tproducts[idx])
                products.append(product)
            elif version == VersionOption.LAST:
                idx = df.iloc[-1]['index']
                pversion = df.iloc[-1]['version']
                product = Product(
                    product_name, pversion, tproducts[idx])
                products.append(product)
            elif version == VersionOption.FIRST:
                idx = df.iloc[0]['index']
                pversion = df.iloc[0]['version']
                product = Product(
                    product_name, pversion, tproducts[idx])
                products.append(product)
            elif version == VersionOption.ALL:
                for idx, row in df.iterrows():
                    idx = row['index']
                    pversion = row['version']
                    product = Product(
                        product_name, pversion, tproducts[idx])
                    products.append(product)
            else:
                msg = 'No VersionOption defined for %s' % version
                raise(AttributeError(msg))

        return products

class Product(object):
    """Class describing a Product from detailed GeoJSON feed.
    """

    def __init__(self, product_name, version, product):
        """Create a product class from product in detailed GeoJSON.
        Args:
            product_name (str): Name of Product (origin, shakemap, etc.)
            version (int): Best guess as to ordinal version of the product.
            product (dict): Product data to be copied from DetailEvent.
        """
        self._product_name = product_name
        self._version = version
        self._product = product.copy()

    def getContentsMatching(self, regexp):
        """Find all contents that match the input regex, shortest to longest.
        Args:
            regexp (str): Regular expression which should match one of the content files
                in the Product.
        Returns:
            list: List of contents matching input regex.
        """
        contents = []
        if not len(self._product['contents']):
            return contents

        for contentkey in self._product['contents'].keys():
            url = self._product['contents'][contentkey]['url']
            parts = urlparse(url)
            fname = parts.path.split('/')[-1]
            if re.search(regexp + '$', fname):
                contents.append(fname)
        return contents

    def __repr__(self):
        ncontents = len(self._product['contents'])
        tpl = (self._product_name, self.source, self.update_time, ncontents)
        return ('Product %s from %s updated %s '
                'containing %i content files.' % tpl)

    def getContentName(self, regexp):
        """Get the shortest filename matching input regular expression.
        For example, if the shakemap product has contents called
        grid.xml and grid.xml.zip, and the input regexp is grid.xml,
        then grid.xml will be matched.
        Args:
            regexp (str): Regular expression to use to search for matching contents.
        Returns:
            str: Shortest file name to match input regexp, or None if
                 no matches found.
        """
        content_name = 'a' * 1000
        found = False
        for contentkey, content in self._product['contents'].items():
            if re.search(regexp + '$', contentkey) is None:
                continue
            url = content['url']
            parts = urlparse(url)
            fname = parts.path.split('/')[-1]
            if len(fname) < len(content_name):
                content_name = fname
                found = True
        if found:
            return content_name
        else:
            return None

    def getContentURL(self, regexp):
        """Get the URL for the shortest filename matching input regular expression.
        For example, if the shakemap product has contents called grid.xml and
        grid.xml.zip, and the input regexp is grid.xml, then grid.xml will be
        matched.
        Args:
            regexp (str): Regular expression to use to search for matching contents.
        Returns:
            str: URL for shortest file name to match input regexp, or
                 None if no matches found.
        """
        content_name = 'a' * 1000
        found = False
        content_url = ''
        for contentkey, content in self._product['contents'].items():
            if re.search(regexp + '$', contentkey) is None:
                continue
            url = content['url']
            parts = urlparse(url)
            fname = parts.path.split('/')[-1]
            if len(fname) < len(content_name):
                content_name = fname
                content_url = url
                found = True
        if found:
            return content_url
        else:
            return None

    def getContent(self, regexp, filename):
        """Download the shortest file name matching the input regular expression.
        Args:
            regexp (str): Regular expression which should match one of the
                content files
                in the Product.
        filename (str): Filename to which content should be downloaded.
        Returns:
            str: The URL from which the content was downloaded.
        Raises:
          Exception: If content could not be downloaded from ComCat
              after two tries.
        """
        data, url = self.getContentBytes(regexp)

        f = open(filename, 'wb')
        f.write(data)
        f.close()

        return url

    def getContentBytes(self, regexp):
        """Return bytes of shortest file name matching input regular expression.
        Args:
            regexp (str): Regular expression which should match one of the
                content files in
                the Product.
        Returns:
            tuple: (array of bytes containing file contents, source url)
                Bytes can be decoded to UTF-8 by the user if file contents are known
                to be ASCII.  i.e.,
                product.getContentBytes('info.json').decode('utf-8')
        Raises:
            Exception: If content could not be downloaded from ComCat
                after two tries.
        """
        content_name = 'a' * 1000
        content_url = None
        for contentkey, content in self._product['contents'].items():
            if re.search(regexp + '$', contentkey) is None:
                continue
            url = content['url']
            parts = urlparse(url)
            fname = parts.path.split('/')[-1]
            if len(fname) < len(content_name):
                content_name = fname
                content_url = url
        if content_url is None:
            raise AttributeError(
                'Could not find any content matching input %s' % regexp)

        try:
            fh = request.urlopen(url, timeout=TIMEOUT)
            data = fh.read()
            fh.close()

        except HTTPError:
            time.sleep(WAITSECS)
            try:
                fh = request.urlopen(url, timeout=TIMEOUT)
                data = fh.read()
                fh.close()
            except Exception:
                raise Exception('Could not download %s from %s.' %
                                (content_name, url))

        return (data, url)

    def hasProperty(self, key):
        """Determine if this Product contains a given property.
        Args:
            key (str): Property to search for.
        Returns:
            bool: Indicates whether that key exists or not.
        """
        if key not in self._product['properties']:
            return False
        return True

    @property
    def preferred_weight(self):
        """The weight assigned to this product by ComCat.
        Returns:
            float: weight assigned to this product by ComCat.
        """
        return self._product['preferredWeight']

    @property
    def source(self):
        """The contributing source for this product.
        Returns:
            str: contributing source for this product.
        """
        return self._product['source']

    @property
    def product_timestamp(self):
        """The timestamp for this product.
        Returns:
            int: The timestamp for this product (effectively used as
                version number by ComCat).
        """
        time_in_msec = self._product['updateTime']
        return time_in_msec

    @property
    def update_time(self):
        """The datetime for when this product was updated.
        Returns:
            datetime: datetime for when this product was updated.
        """
        time_in_msec = self._product['updateTime']
        time_in_sec = time_in_msec // 1000
        msec = time_in_msec - (time_in_sec * 1000)
        dtime = datetime.utcfromtimestamp(time_in_sec)
        dt = timedelta(milliseconds=msec)
        dtime = dtime + dt
        return dtime

    @property
    def version(self):
        """The best guess for the ordinal version number of this product.
        Returns:
            int: best guess for the ordinal version number of this product.
        """
        return self._version

    @property
    def properties(self):
        """List of product properties.
        Returns:
            list: List of product properties (retrievable from object with [] operator).
        """
        return list(self._product['properties'].keys())

    @property
    def contents(self):
        """List of product properties.
        Returns:
            list: List of product properties (retrievable with getContent() method).
        """
        return list(self._product['contents'].keys())

    def __getitem__(self, key):
        """Extract Product property using the [] operator.
        Args:
            key (str): Property to extract.
        Returns:
            str: Desired property.
        """
        if key not in self._product['properties']:
            msg = 'No property %s found in %s product.' % (
                key, self._product_name)
            raise AttributeError(msg)
        return self._product['properties'][key]

def _get_moment_tensor_info(tensor, get_angles=False,
                            get_moment_supplement=False):
    """Internal - gather up tensor components and focal mechanism angles.
    """
    msource = tensor['eventsource']
    if tensor.hasProperty('derived-magnitude-type'):
        msource += '_' + tensor['derived-magnitude-type']
    elif tensor.hasProperty('beachball-type'):
        btype = tensor['beachball-type']
        if btype.find('/') > -1:
            btype = btype.split('/')[-1]
        msource += '_' + btype

    edict = OrderedDict()
    edict['%s_mrr' % msource] = float(tensor['tensor-mrr'])
    edict['%s_mtt' % msource] = float(tensor['tensor-mtt'])
    edict['%s_mpp' % msource] = float(tensor['tensor-mpp'])
    edict['%s_mrt' % msource] = float(tensor['tensor-mrt'])
    edict['%s_mrp' % msource] = float(tensor['tensor-mrp'])
    edict['%s_mtp' % msource] = float(tensor['tensor-mtp'])
    if get_angles and tensor.hasProperty('nodal-plane-1-strike'):
        edict['%s_np1_strike' % msource] = tensor['nodal-plane-1-strike']
        edict['%s_np1_dip' % msource] = tensor['nodal-plane-1-dip']
        if tensor.hasProperty('nodal-plane-1-rake'):
            edict['%s_np1_rake' % msource] = tensor['nodal-plane-1-rake']
        else:
            edict['%s_np1_rake' % msource] = tensor['nodal-plane-1-slip']
        edict['%s_np2_strike' % msource] = tensor['nodal-plane-2-strike']
        edict['%s_np2_dip' % msource] = tensor['nodal-plane-2-dip']
        if tensor.hasProperty('nodal-plane-2-rake'):
            edict['%s_np2_rake' % msource] = tensor['nodal-plane-2-rake']
        else:
            edict['%s_np2_rake' % msource] = tensor['nodal-plane-2-slip']

    if get_moment_supplement:
        if tensor.hasProperty('derived-latitude'):
            edict['%s_derived_latitude' % msource] = float(
                tensor['derived-latitude'])
            edict['%s_derived_longitude' % msource] = float(
                tensor['derived-longitude'])
            edict['%s_derived_depth' % msource] = float(
                tensor['derived-depth'])
        if tensor.hasProperty('percent-double-couple'):
            edict['%s_percent_double_couple' % msource] = float(
                tensor['percent-double-couple'])
        if tensor.hasProperty('sourcetime-duration'):
            edict['%s_sourcetime_duration' % msource] = float(
                tensor['sourcetime-duration'])

    return edict

def _get_focal_mechanism_info(focal):
    """Internal - gather up focal mechanism angles.
    """
    msource = focal['eventsource']
    eventid = msource + focal['eventsourcecode']
    edict = OrderedDict()
    try:
        edict['%s_np1_strike' % msource] = focal['nodal-plane-1-strike']
    except Exception:
        sys.stderr.write(
            'No focal angles for %s in detailed geojson.\n' % eventid)
        return edict
    edict['%s_np1_dip' % msource] = focal['nodal-plane-1-dip']
    if focal.hasProperty('nodal-plane-1-rake'):
        edict['%s_np1_rake' % msource] = focal['nodal-plane-1-rake']
    else:
        edict['%s_np1_rake' % msource] = focal['nodal-plane-1-slip']
    edict['%s_np2_strike' % msource] = focal['nodal-plane-2-strike']
    edict['%s_np2_dip' % msource] = focal['nodal-plane-2-dip']
    if focal.hasProperty('nodal-plane-2-rake'):
        edict['%s_np2_rake' % msource] = focal['nodal-plane-2-rake']
    else:
        edict['%s_np2_rake' % msource] = focal['nodal-plane-2-slip']
    return edict

def get_event_by_id(eventid, catalog=None,
                    includedeleted=False,
                    includesuperseded=False,
                    host=None):
    """Search the ComCat database for an event matching the input event id.
    This search function is a wrapper around the ComCat Web API described here:
    https://earthquake.usgs.gov/fdsnws/event/1/
    Some of the search parameters described there are NOT implemented here, usually because they do not
    apply to GeoJSON search results, which we are getting here and parsing into Python data structures.
    This function returns a DetailEvent object, described elsewhere in this package.
    Usage:

    Args:
        eventid (str): Select a specific event by ID; event identifiers are data center specific.
        includesuperseded (bool):
            Specify if superseded products should be included. This also includes all
            deleted products, and is mutually exclusive to the includedeleted parameter.
        includedeleted (bool): Specify if deleted products should be incuded.
        host (str): Replace default ComCat host (earthquake.usgs.gov) with a custom host.
    Returns: DetailEvent object.
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

    event = _search(**newargs)  # this should be a DetailEvent
    return event