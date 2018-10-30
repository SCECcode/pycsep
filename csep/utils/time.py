import datetime
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR

def epoch_time_to_utc_datetime(epoch_time_milli):
    """
    Accepts an epoch_time in milliseconds the UTC timezone and returns a python datetime object.

    :param epoch_time: epoch_time in UTC timezone
    :type epoch_time: float
    """
    epoch_time = epoch_time_milli / 1000
    dt = datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc)
    return dt

def datetime_to_utc_epoch(datetime):
    pass


def timedelta_from_years(time_in_years):
    """
    Returns python datetime.timedelta object based on the astronomical year in seconds.

    :params time_in_years: positive fraction of years 0 <= time_in_years
    :type time_in_years: float
    """
    if time_in_years < 0:
        raise ValueError("time_in_years must be greater than zero.")

    seconds = SECONDS_PER_ASTRONOMICAL_YEAR * time_in_years
    time_delta = datetime.timedelta(seconds=seconds)
    return time_delta
