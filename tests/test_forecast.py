import os
import unittest
import numpy
from csep import load_catalog_forecast


def get_test_catalog_root():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'test_ascii_catalogs')
    return data_dir


class TestCatalogForecastCreation(unittest.TestCase):

    def test_all_present(self):
        fname = os.path.join(get_test_catalog_root(), 'all_present.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        self.assertEqual(10, test_fore.n_cat)
        self.assertEqual(10, total_event_count)

    def test_ascii_load_all_empty(self):
        fname = os.path.join(get_test_catalog_root(), 'all_empty.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        self.assertEqual(0, total_event_count)
        self.assertEqual(10, test_fore.n_cat)
        numpy.testing.assert_array_equal([cat.catalog_id for cat in test_fore], numpy.arange(10))

    def test_ascii_load_all_empty_verbose(self):
        fname = os.path.join(get_test_catalog_root(), 'all_empty_verbose.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        self.assertEqual(0, total_event_count)
        self.assertEqual(10, test_fore.n_cat)
        numpy.testing.assert_array_equal([cat.catalog_id for cat in test_fore], numpy.arange(10))

    def test_ascii_load_last_empty(self):
        fname = os.path.join(get_test_catalog_root(), 'last_empty.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        self.assertEqual(1, total_event_count)
        self.assertEqual(10, test_fore.n_cat)
        numpy.testing.assert_array_equal([cat.catalog_id for cat in test_fore], numpy.arange(10))

    def test_ascii_some_missing(self):
        fname = os.path.join(get_test_catalog_root(), 'some_empty.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        print([(cat.event_count, cat.catalog_id) for cat in test_fore])
        self.assertEqual(2, total_event_count)
        self.assertEqual(10, test_fore.n_cat)
        numpy.testing.assert_array_equal([cat.catalog_id for cat in test_fore], numpy.arange(10))

    def test_ascii_some_missing_verbose(self):
        fname = os.path.join(get_test_catalog_root(), 'all_present.csv')
        test_fore = load_catalog_forecast(fname)
        total_event_count = numpy.array([cat.event_count for cat in test_fore]).sum()
        print([(cat.event_count, cat.catalog_id) for cat in test_fore])
        self.assertEqual(10, total_event_count)
        self.assertEqual(10, test_fore.n_cat)
        numpy.testing.assert_array_equal([cat.catalog_id for cat in test_fore], numpy.arange(10))

    def test_get_event_counts(self):
        fname = os.path.join(get_test_catalog_root(), 'all_present.csv')
        test_fore = load_catalog_forecast(fname)
        numpy.testing.assert_array_equal(numpy.ones(10), test_fore.get_event_counts())

    def test_multiple_iterations(self):
        fname = os.path.join(get_test_catalog_root(), 'all_present.csv')
        test_fore = load_catalog_forecast(fname)
        ec1 = [cat.event_count for cat in test_fore]
        ec2 = [cat.event_count for cat in test_fore]
        numpy.testing.assert_array_equal(ec1, ec2)

if __name__ == '__main__':
    unittest.main()
