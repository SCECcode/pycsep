import unittest

import os.path

import numpy

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Testing', 'JMA-catalog')
    return data_dir

def test_JmaCsvCatalog_loading():
    datadir = get_datadir()
    csv_file = os.path.join(datadir, 'test.csv')
    from csep.core.catalogs import JmaCsvCatalog
    _JmaCsvCatalogObject = JmaCsvCatalog(filename = csv_file)
    _JmaCsvCatalogObject.load_catalog()

    assert len(_JmaCsvCatalogObject.catalog) == 22284, 'invalid number of events in catalog object'

    _dummy = _JmaCsvCatalogObject.get_magnitudes()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    _dummy = _JmaCsvCatalogObject.get_depths()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    _dummy = _JmaCsvCatalogObject.get_longitudes()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    _dummy = _JmaCsvCatalogObject.get_latitudes()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    _dummy = _JmaCsvCatalogObject.get_epoch_times()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    _dummy = _JmaCsvCatalogObject.get_datetimes()
    assert len(_dummy) == len(_JmaCsvCatalogObject.catalog)

    # assert (d[0].timestamp() * 1000.) == c.catalog['timestamp'][0]

    _datetimes = numpy.ndarray(len(_JmaCsvCatalogObject.catalog), dtype='<i8')
    _datetimes.fill(numpy.nan)

    for _idx, _val in enumerate(_dummy):
        _datetimes[_idx] = round(1000. * _val.timestamp())

    numpy.testing.assert_allclose(_datetimes, _JmaCsvCatalogObject.catalog['timestamp'],
                                  err_msg='timestamp mismatch',
                                  verbose=True, rtol=0, atol=0)

    _ZMAPCatalogObject = _JmaCsvCatalogObject._get_csep_format()
    assert len(_ZMAPCatalogObject.catalog) == len(_JmaCsvCatalogObject.catalog)
    numpy.testing.assert_allclose(_ZMAPCatalogObject.get_magnitudes(), _JmaCsvCatalogObject.get_magnitudes(),
                                  err_msg='ZMAPCatalog magnitudes differ from JmaCsvCatalog magnitudes',
                                  verbose=True, rtol=1e-06, atol=0)