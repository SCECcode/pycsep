import os.path
import vcr
from datetime import datetime
from csep.utils.readers import _query_gcmt
import unittest

root_dir = os.path.dirname(os.path.abspath(__file__))


def gcmt_dir():
    data_dir = os.path.join(root_dir, 'artifacts', 'gCMT')
    return data_dir


class TestCatalogGetter(unittest.TestCase):

    def test_gcmt_search(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_search.yaml')
        with vcr.use_cassette(tape_file):
            # Maule, Chile
            eventlist = \
                _query_gcmt(start_time=datetime(2010, 2, 26),
                            end_time=datetime(2010, 3, 2),
                            min_magnitude=7)[0]
            event = eventlist
            assert event[0] == '2844986'

    def test_isc_gcmt_summary(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_summary.yaml')
        with vcr.use_cassette(tape_file):
            eventlist = \
                _query_gcmt(start_time=datetime(2010, 2, 26),
                            end_time=datetime(2010, 3, 2),
                            min_magnitude=7)
            event = eventlist[0]
            cmp = "('2844986', 1267252514000, -35.98, -73.15, 23.2, 8.8)"
            assert str(event) == cmp
            assert event[0] == '2844986'
            assert datetime.fromtimestamp(
                event[1] / 1000.) == datetime.fromtimestamp(1267252514)
            assert event[2] == -35.98
            assert event[3] == -73.15
            assert event[4] == 23.2
            assert event[5] == 8.8
