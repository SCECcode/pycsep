import unittest
import csep
from csep.utils.time_utils import datetime_to_utc_epoch
import os.path
import numpy

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'JMA-observed_catalog')
    return data_dir

def test_JmaCsvCatalog_loading():
    datadir = get_datadir()
    csv_file = os.path.join(datadir, 'test.csv')

    test_catalog = csep.load_catalog(csv_file, type='jma-csv')

    assert len(test_catalog.catalog) == 22284, 'invalid number of events in observed_catalog object'

    _dummy = test_catalog.get_magnitudes()
    assert len(_dummy) == len(test_catalog.catalog)

    _dummy = test_catalog.get_depths()
    assert len(_dummy) == len(test_catalog.catalog)

    _dummy = test_catalog.get_longitudes()
    assert len(_dummy) == len(test_catalog.catalog)

    _dummy = test_catalog.get_latitudes()
    assert len(_dummy) == len(test_catalog.catalog)

    _dummy = test_catalog.get_epoch_times()
    assert len(_dummy) == len(test_catalog.catalog)

    _dummy = test_catalog.get_datetimes()
    assert len(_dummy) == len(test_catalog.catalog)

    # assert (d[0].timestamp() * 1000.) == c.observed_catalog['timestamp'][0]

    _datetimes = numpy.ndarray(test_catalog.event_count, dtype='<i8')
    # _datetimes.fill(numpy.nan)

    for _idx, _val in enumerate(_dummy):
        _datetimes[_idx] = datetime_to_utc_epoch(_val)

    numpy.testing.assert_allclose(_datetimes, test_catalog.catalog['origin_time'],
                                  err_msg='timestamp mismatch',
                                  verbose=True,
                                  rtol=1e-3, atol=0)
