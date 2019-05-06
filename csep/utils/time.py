import datetime
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR

def epoch_time_to_utc_datetime(epoch_time_milli):
    """
    Accepts an epoch_time in milliseconds the UTC timezone and returns a python datetime object.

    See https://docs.python.org/3/library/datetime.html#datetime.datetime.fromtimestamp for information
    about how timezones are handled with this function.

    :param epoch_time: epoch_time in UTC timezone in milliseconds
    :type epoch_time: float
    """
    if epoch_time_milli is None:
        return epoch_time_milli
    epoch_time = epoch_time_milli / 1000
    dt = datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc)
    return dt

def datetime_to_utc_epoch(dt):
    """
    Converts python datetime.datetime into epoch_time in milliseconds.


    Args:
        dt (datetime.datetime): python datetime object, should be naive.
    """
    if dt is None:
        return dt
    now = datetime.datetime.utcnow()
    epoch = datetime.datetime(1970,1,1)
    epoch_time_seconds = (dt - epoch).total_seconds()
    return 1000.0 * epoch_time_seconds


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

def zmap_time_to_datetime(year=None, month=None, day=None,
                          hour=None, minute=None, second=None):
    """
    Converts time in ZMAP format into python datetime object

    Returns:
        datetime object
    """
    pass

def strptime_to_utc_datetime(time_string, format):
    """
    Converts time_string with format into time-zone aware datetime object in the UTC timezone.

    Note:
        If the time_string is not in UTC time, it will be converted into UTC timezone.

    Args:
        time_string (str): string representation of datetime
        format (str): format of time_string

    Returns:
        datetime.datetime: timezone aware (utc) object from time_string
    """
    dt = datetime.datetime.strptime(time_string, format).replace(tzinfo=datetime.timezone.utc)
    return dt
