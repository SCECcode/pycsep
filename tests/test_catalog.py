import copy
import unittest
import os
import itertools

import numpy

import csep
from csep.core import regions, forecasts
from csep.utils.time_utils import strptime_to_utc_epoch, strptime_to_utc_datetime
from csep.core.catalogs import CSEPCatalog, AbstractBaseCatalog
from csep.core.regions import CartesianGrid2D, compute_vertices
from csep.models import Polygon


def comcat_path():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Comcat',
                            'test_catalog.csv')
    return data_dir

class MockCatalog(AbstractBaseCatalog):

    def __init__(self):
        super().__init__()

    def get_number_of_events(self):
        return len(self.get_latitudes())

    def get_longitudes(self):
        return numpy.array([0.05, 0.15])

    def get_latitudes(self):
        return numpy.array([0.15, 0.05])

class CatalogFiltering(unittest.TestCase):
    def setUp(self):

        # create some arbitrary grid
        self.nx = 8
        self.ny = 10
        self.dh = 1
        x_points = numpy.arange(1.5, self.nx) * self.dh
        y_points = numpy.arange(1.5, self.ny) * self.dh

        # spatial grid starts at (1.5, 1.5); so the event at (1, 1) should be removed.
        self.origins = list(itertools.product(x_points, y_points))
        self.num_nodes = len(self.origins)
        self.cart_grid = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(self.origins, self.dh)], self.dh)

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
        numpy.testing.assert_array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_separately(self):
        # Filter together
        start_epoch = strptime_to_utc_epoch('2009-07-01 00:00:00.0')
        end_epoch = strptime_to_utc_epoch('2010-07-01 00:00:00.0')
        filters = [f'origin_time >= {start_epoch}', f'origin_time < {end_epoch}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        # Filter separately
        for i in filters:
            test_cat.filter(i)

        numpy.testing.assert_array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_with_datetime_list(self):
        start_dt = strptime_to_utc_datetime('2009-07-01 00:00:00.0')
        end_dt = strptime_to_utc_datetime('2010-07-01 00:00:00.0')
        filters = [f'datetime >= {start_dt}', f'datetime < {end_dt}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        test_cat.filter(filters)
        numpy.testing.assert_array_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_with_datetime_in_place_list(self):
        start_dt = strptime_to_utc_datetime('2009-07-01 00:00:00.0')
        end_dt = strptime_to_utc_datetime('2010-07-01 00:00:00.0')
        filters = [f'datetime > {start_dt}', f'datetime < {end_dt}']  # should return only event 2
        test_cat = copy.deepcopy(self.test_cat1)
        test_cat = test_cat.filter(filters, in_place=True)
        numpy.testing.assert_equal(numpy.array([b'2'], dtype='S256'), test_cat.get_event_ids())

    def test_filter_with_datetime(self):
        end_dt = strptime_to_utc_datetime('2010-07-01 00:00:00.0')
        filters = f'datetime < {end_dt}'  # should return only event 1 and 2
        test_cat = copy.deepcopy(self.test_cat1)
        filtered_test_cat = test_cat.filter(filters, in_place=False)
        numpy.testing.assert_equal(numpy.array([b'1', b'2'], dtype='S256').T, filtered_test_cat.get_event_ids())

    def test_filter_spatial(self):

        test_cat = copy.deepcopy(self.test_cat1)
        filtered_test_cat = test_cat.filter_spatial(region=self.cart_grid)
        numpy.testing.assert_equal(numpy.array([b'2', b'3'], dtype='S256').T, filtered_test_cat.get_event_ids())

    def test_filter_spatial_in_place_return(self):
        test_cat = copy.deepcopy(self.test_cat1)
        filtered_test_cat = test_cat.filter_spatial(region=self.cart_grid, in_place=False)
        numpy.testing.assert_array_equal(filtered_test_cat.region.midpoints(), test_cat.region.midpoints())
        numpy.testing.assert_array_equal(filtered_test_cat.region.origins(), test_cat.region.origins())
        numpy.testing.assert_array_equal(filtered_test_cat.region.bbox_mask, test_cat.region.bbox_mask)
        numpy.testing.assert_array_equal(filtered_test_cat.region.idx_map, test_cat.region.idx_map)

    def test_catalog_binning_and_filtering_acceptance(self):
        # create space-magnitude region
        region = regions.create_space_magnitude_region(
            regions.california_relm_region(),
            regions.magnitude_bins(4.5, 10.05, 0.1)
        )

        # read catalog
        comcat = csep.load_catalog(comcat_path(), region=region).filter(f"magnitude >= 4.5")

        # create data set from data set
        d = forecasts.MarkedGriddedDataSet(
            data=comcat.spatial_magnitude_counts(),
            region=comcat.region,
            magnitudes=comcat.region.magnitudes
        )

        for idm, m_min in enumerate(d.magnitudes):
            # catalog filtered cumulative
            print(m_min)
            c = comcat.filter([f'magnitude >= {m_min}'], in_place=False)
            # catalog filtered incrementally
            c_int = comcat.filter([f'magnitude >= {m_min}', f'magnitude < {m_min + 0.1}'], in_place=False)
            # sum from overall data set
            gs = d.data[:, idm:].sum()
            # incremental counts
            gs_int = d.data[:, idm].sum()
            # event count from filtered catalog and events in binned data should be the same
            numpy.testing.assert_equal(gs, c.event_count)
            numpy.testing.assert_equal(gs_int, c_int.event_count)

    def test_bin_spatial_counts(self):
        """ this will test both good and bad points within the region.

        1) 2 inside the domain
        2) outside the bbox but not in domain
        3) completely outside the domain

        we will check that only 1 event is placed in the grid and ensure its location is correct.
        """
        # create some arbitrary grid
        nx = 8
        ny = 10
        dh = 0.1
        x_points = numpy.arange(nx) * dh
        y_points = numpy.arange(ny) * dh
        origins = list(itertools.product(x_points, y_points))
        # grid is missing first and last block
        origins.pop(0)
        origins.pop(-1)
        cart_grid = CartesianGrid2D(
            [Polygon(bbox) for bbox in compute_vertices(origins, dh)],
            dh
        )
        catalog = MockCatalog()
        catalog.region = cart_grid
        test_result = catalog.spatial_counts()

        # we have tested 2 inside the domain
        self.assertEqual(numpy.sum(test_result), 2)
        # we know that (0.05, 0.15) corresponds to index 0
        self.assertEqual(test_result[0], 1)
        # we know that (0.15, 0.05) corresponds too index 9
        self.assertEqual(test_result[9], 1)

    def test_bin_probability(self):
        # we have tested 2 inside the domain
        nx = 8
        ny = 10
        dh = 0.1
        x_points = numpy.arange(nx) * dh
        y_points = numpy.arange(ny) * dh
        origins = list(itertools.product(x_points, y_points))
        # grid is missing first and last block
        origins.pop(0)
        origins.pop(-1)
        cart_grid = CartesianGrid2D(
            [Polygon(bbox) for bbox in compute_vertices(origins, dh)],
            dh
        )
        catalog = MockCatalog()
        catalog.region = cart_grid
        test_result = catalog.spatial_event_probability()
        self.assertEqual(numpy.sum(test_result), 2)

        # we know that (0.05, 0.15) corresponds to index 0 from the above test
        self.assertEqual(test_result[0], 1)
        self.assertEqual(test_result[9], 1)

if __name__ == '__main__':
    unittest.main()
