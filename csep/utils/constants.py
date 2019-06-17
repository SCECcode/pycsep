# saving year as astronomical year
# calculated as 365.25*24*60*60

from enum import Enum

# Time Constants
SECONDS_PER_ASTRONOMICAL_YEAR = 31557600
SECONDS_PER_DAY = 60*60*24
SECONDS_PER_WEEK = SECONDS_PER_DAY*7
SECONDS_PER_MONTH = SECONDS_PER_WEEK*4


class JobStatus(str, Enum):
    FAILED = 'failed'
    UNPREPARED = 'unprepared'
    COMPLETE = 'complete'
    PREPARED = 'prepared'
    SUBMITTED = 'submitted'
    RUNNING = 'running'
