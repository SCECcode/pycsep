from datetime import datetime
import os.path
import vcr
from csep.utils.comcat import search

HOST = 'webservices.rm.ingv.it'


def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'BSI')
    return data_dir


def test_search():
    datadir = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_search.yaml')
    with vcr.use_cassette(tape_file):
        # L'Aquila
        eventlist = search(starttime=datetime(2009, 4, 6, 0, 0, 0),
                           endtime=datetime(2009, 4, 7, 0, 0, 0),
                           minmagnitude=5.5, host=HOST, limit=15000, offset=0)
        event = eventlist[0]
        assert event.id == 1895389


def test_summary():
    datadir = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_summary.yaml')
    with vcr.use_cassette(tape_file):
        eventlist = search(starttime=datetime(2009, 4, 6, 0, 0, 0),
                           endtime=datetime(2009, 4, 7, 0, 0, 0),
                           minmagnitude=5.5, host=HOST, limit=15000, offset=0)
        event = eventlist[0]
        cmp = '1895389 2009-04-06 01:32:40.400000 (42.342,13.380) 8.3 km M6.1'
        assert str(event) == cmp
        assert event.id == 1895389
        assert event.time == datetime(2009, 4, 6, 1, 32, 40, 400000)
        assert event.latitude == 42.342
        assert event.longitude == 13.380
        assert event.depth == 8.3
        assert event.magnitude == 6.1
