#! /usr/bin/env python

## This script checks for COMCAT_REPO env_var is set and points to a writeable directory.
## For example, you can set this script up to a cronjob to recovery the ComCat catalog daily.

import datetime
import time
import os

from csep.core.catalogs import ComcatCatalog
from csep.core.repositories import FileSystem

# query dates
start = datetime.datetime(1985,1,1,0,0,0,0)
now = datetime.datetime.now()

t0 = time.time()
print('Fetching ComCat catalog...')
try:
    comcat = ComcatCatalog(start_time = start, end_time = now,
                           min_magnitude = 2.5, max_magnitude = 10.0,
                           min_latitude = 31.5, max_latitude = 43.0,
                           min_longitude = -125.4, max_longitude = -113.1, compute_stats=False, name='ComCat')
    t1 = time.time()
except:
    print('Trying to connect with Comcat again...')
    comcat = ComcatCatalog(start_time = start, end_time = now,
                           min_magnitude = 2.5, max_magnitude = 10.0,
                           min_latitude = 31.5, max_latitude = 43.0,
                           min_longitude = -125.4, max_longitude = -113.1, compute_stats=False, name='ComCat')
    t1 = time.time()

print("Fetched Comcat catalog in {} seconds.\n".format(t1 - t0))
print("Downloaded Comcat Catalog with following parameters")
print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
print("Min Magnitude: {}\n".format(comcat.min_magnitude))

# file paths
try:
    base_dir = os.environ['COMCAT_REPO']
except:
    base_dir = '.'
fname = os.path.join(base_dir, now.strftime('%Y-%m-%d-comcat.json'))
repo = FileSystem(url=fname)

print('Writing catalog to Json format..')
repo.save(comcat.to_dict())