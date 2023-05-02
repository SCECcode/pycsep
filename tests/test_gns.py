from datetime import datetime
import os.path
from csep.utils.geonet import gns_search
import vcr

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'geonet')
    return data_dir

def test_search():
    datadir = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_search.yaml')
    with vcr.use_cassette(tape_file):
        eventlist = gns_search(starttime=datetime(2023, 2, 1, 0, 0, 0),
                        endtime=datetime(2023, 2, 10, 0, 0, 0),
                        minmagnitude=4)
        event = eventlist[0]
        assert event.id == '2023p087955'


def test_summary():
    datadir = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_summary.yaml')
    with vcr.use_cassette(tape_file):
        eventlist = gns_search(starttime=datetime(2023, 2, 1, 0, 0, 0),
                            endtime=datetime(2023, 2, 10, 0, 0, 0),
                            minmagnitude=4)
        event = eventlist[0]
        assert str(event) == "2023p087955 2023-02-02 13:02:43.493000 (-37.587,175.705) 6.0 km M4.8"
        assert event.id == "2023p087955"
        assert event.time == datetime(2023, 2, 2, 13, 2, 43, 493000)
        assert round(event.latitude,3) == -37.587
        assert round(event.longitude,3) == 175.705
        assert round(event.depth) == 6
        assert round(event.magnitude,1) == 4.8
