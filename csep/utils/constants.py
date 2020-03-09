# saving year as astronomical year
# calculated as 365.25*24*60*60

from enum import Enum

# Time Constants
import numpy

SECONDS_PER_ASTRONOMICAL_YEAR = 31557600
SECONDS_PER_DAY = 60*60*24
SECONDS_PER_HOUR = 60*60
SECONDS_PER_WEEK = SECONDS_PER_DAY*7
SECONDS_PER_MONTH = SECONDS_PER_WEEK*4

MW_5_EQS_PER_YEAR = 10


class JobStatus(str, Enum):
    FAILED = 'failed'
    UNPREPARED = 'unprepared'
    COMPLETE = 'complete'
    PREPARED = 'prepared'
    SUBMITTED = 'submitted'
    RUNNING = 'running'

# Magnitude Bins
min_mw = 2.5
max_mw = 8.95
dmw = 0.1
CSEP_MW_BINS = numpy.arange(min_mw, max_mw+dmw/2, dmw)
