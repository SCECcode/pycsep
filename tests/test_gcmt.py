import os.path
import vcr
from datetime import datetime
from csep.utils.readers import _query_gcmt
import unittest

root_dir = os.path.dirname(os.path.abspath(__file__))


def gcmt_dir():
    data_dir = os.path.join(root_dir, '../artifacts', 'gCMT')
    return data_dir


class TestCatalogGetter(unittest.TestCase):

    def test_isc_gcmt_search(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_search.yaml')
        with vcr.use_cassette(tape_file):
            # Maule, Chile
            eventlist = \
                _query_gcmt(start_year=2010, start_month=2, start_day=26,
                            end_year=2010, end_month=2, end_day=28,
                            min_mag=8.5)[0]
            event = eventlist[0]
            assert event[0] == '14340585'

    def test_isc_gcmt_summary(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_summary.yaml')
        with vcr.use_cassette(tape_file):
            eventlist = \
                _query_gcmt(start_year=2010, start_month=2, start_day=26,
                            end_year=2010, end_month=2, end_day=28,
                            min_mag=8.5)[0]
            event = eventlist[0]
            cmp = "('14340585', 1267252513600, -35.98, -73.15, 23.2, 8.78)"
            assert str(event) == cmp
            assert event[0] == '14340585'
            assert datetime.fromtimestamp(
                event[1] / 1000.) == datetime.fromtimestamp(1267252513.600)
            assert event[2] == -35.98
            assert event[3] == -73.15
            assert event[4] == 23.2
            assert event[5] == 8.78
