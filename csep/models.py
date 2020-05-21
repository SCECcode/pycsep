from csep.utils.time_utils import datetime_to_utc_epoch, epoch_time_to_utc_datetime

class Simulation:
    """
    View of CSEP Experiment. Contains minimal information required to perform evaluations of
    CSEP Forecasts
    """
    def __init__(self, filename='', min_mw=2.5, start_time=-1, sim_type='', name=''):
        self.filename = filename
        self.min_mw = min_mw
        self.start_time = start_time
        self.sim_type = sim_type
        self.name = name


class Event:
    def __init__(self, id=None, magnitude=None, latitude=None, longitude=None, time=None):
        self.id = id
        self.magnitude = magnitude
        self.latitude = latitude
        self.longitude = longitude
        self.time = time

    @classmethod
    def from_dict(cls, adict):
        return cls(id=adict['id'],
                  magnitude=adict['magnitude'],
                  latitude=adict['latitude'],
                  longitude=adict['longitude'],
                  time=epoch_time_to_utc_datetime(adict['time']))

    def to_dict(self):
        adict = {
            'id': self.id,
            'magnitude': self.magnitude,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'time': datetime_to_utc_epoch(self.time)
        }
        return adict