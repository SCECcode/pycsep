import unittest

import os.path

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Testing', 'JMA-catalog')
    return data_dir

def test_catalog_loading():
    datadir = get_datadir()
    csv_file = os.path.join(datadir, 'test.csv')
    from csep.core.catalogs import JmaCsvCatalog
    c = JmaCsvCatalog(filename = csv_file)
    c.load_catalog()

    assert len(c.catalog) == 22284, 'invalid number of events in catalog object'

    d = c.get_magnitudes()
    assert len(d) == len(c.catalog)

    d = c.get_longitudes()
    assert len(d) == len(c.catalog)

    d = c.get_latitudes()
    assert len(d) == len(c.catalog)

    d = c.get_epoch_times()
    assert len(d) == len(c.catalog)

    d = c.get_datetimes()
    assert len(d) == len(c.catalog)

    assert (d[0].timestamp() * 1000.) == c.catalog['timestamp'][0]