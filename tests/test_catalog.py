import copy
import unittest

import numpy

from csep.utils.time_utils import strptime_to_utc_epoch, strptime_to_utc_datetime
from csep.core.catalogs import CSEPCatalog


class CatalogFiltering(unittest.TestCase):
    def setUp(self):

        # define dummy cat
        date1 = strptime_to_utc_epoch('2009-01-01 00:00:00.0000')
        date2 = strptime_to_utc_epoch('2010-01-01 00:00:00.0000')
        date3 = strptime_to_utc_epoch('2011-01-01 00:00:00.0000')
        catalog = [(b'1', date1, 1.0, 1.0, 1.0, 1.0),
                   (b'2', date2, 2.0, 2.0, 2.0, 2.0),
                   (b'3', date3, 3.0, 3.0, 3.0, 3.0)]
        self.test_cat1 = CSEPCatalog(data=catalog)

    def test_filter_with_list(self):
        # Filter together
        start_epoch = strptime_to_utc_epoch('2009-07-01 00:00:00.0')
        end_epoch = strptime_to_utc_epoch('2010-07-01 00:00:00.0')
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        test_cat.filter(filters)
        # Filter together
        numpy.array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_separately(self):
        # Filter together
        start_epoch = strptime_to_utc_epoch('2009-07-01 00:00:00.0')
        end_epoch = strptime_to_utc_epoch('2010-07-01 00:00:00.0')
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        # Filter separately
        for i in filters:
            test_cat.filter(i)

        numpy.array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_with_datetime(self):
        start_dt = strptime_to_utc_datetime('2009-07-01 00:00:00.0')
        end_dt = strptime_to_utc_datetime('2010-07-01 00:00:00.0')
        filters = [f'datetime >= {start_dt}', f'datetime < {end_dt}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        test_cat.filter(filters)
        numpy.array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())
if __name__ == '__main__':
    unittest.main()
