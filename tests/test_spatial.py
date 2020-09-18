import unittest
import itertools
import pytest

import numpy

from csep.core.regions import CartesianGrid2D, compute_vertex, compute_vertices, _bin_catalog_spatio_magnitude_counts, \
    _bin_catalog_spatial_counts, _bin_catalog_probability, Polygon


class TestPolygon(unittest.TestCase):

    def setUp(self):
        dh = 1
        origin = (0,0)
        self.polygon = Polygon(compute_vertex(origin, dh))

    def test_object_creation(self):
        self.assertTupleEqual(self.polygon.origin,(0,0))
        numpy.testing.assert_allclose(self.polygon.points,[(0,0),(0,1),(1,1),(1,0)])

    def test_contains_inside(self):
        test_point = (0.5, 0.5)
        self.assertTrue(self.polygon.contains(test_point))

    def test_contains_outside(self):
        test_point = (-0.5, -0.5)
        self.assertFalse(self.polygon.contains(test_point))

    def test_compute_centroid(self):
        expected = (0.5, 0.5)
        numpy.testing.assert_almost_equal(self.polygon.centroid(), expected)

class TestCartesian2D(unittest.TestCase):

    def setUp(self):

        # create some arbitrary grid
        self.nx = 8
        self.ny = 10
        self.dh = 0.1
        x_points = numpy.arange(self.nx)*self.dh
        y_points = numpy.arange(self.ny)*self.dh
        self.origins = list(itertools.product(x_points, y_points))
        # grid is missing first and last block
        self.origins.pop(0)
        self.origins.pop(-1)
        self.num_nodes = len(self.origins)
        # this is kinda ugly, maybe we want to create this in a different way, class method?
        self.cart_grid = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(self.origins, self.dh)], self.dh)

    def test_object_creation(self):
        self.assertEqual(self.cart_grid.dh, self.dh, 'dh did not get initialized properly')
        self.assertEqual(self.cart_grid.num_nodes, self.num_nodes, 'num nodes is not correct')

    def test_xs_and_xy_correct(self):
        numpy.testing.assert_allclose(self.cart_grid.xs, numpy.arange(0,self.nx)*self.dh)
        numpy.testing.assert_allclose(self.cart_grid.ys, numpy.arange(0,self.ny)*self.dh)

    def test_bitmask_indices_mapping(self):
        test_idx = self.cart_grid.idx_map[1,0]
        numpy.testing.assert_allclose(test_idx, 0, err_msg='mapping for first polygon index (good) not correct')

        test_idx = self.cart_grid.idx_map[0,1]
        numpy.testing.assert_allclose(test_idx, 9, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[2,0]
        numpy.testing.assert_allclose(test_idx, 1, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[0,2]
        numpy.testing.assert_allclose(test_idx, 19, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[-1,-1]
        numpy.testing.assert_allclose(test_idx, numpy.nan, err_msg='mapping for last index (bad) not correct.')

        test_idx = self.cart_grid.idx_map[0,0]
        numpy.testing.assert_allclose(test_idx, numpy.nan, err_msg='mapping for first index (bad) not correct.')

    def test_domain_mask(self):
        test_flag = self.cart_grid.mask[0,0]
        self.assertEqual(test_flag, 1)

        test_flag = self.cart_grid.mask[-1,1]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.mask[1, -1]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.mask[2,2]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.mask[-1, -1]
        self.assertEqual(test_flag, 1)

    def test_get_index_of_outside_bbox(self):
        test = (-0.05, -0.05)
        with pytest.raises(ValueError):
            self.cart_grid.get_index_of([test[0]], [test[1]])

    def test_get_index_of_inside_bbox_but_masked(self):
        test = (0.05, 0.05)
        with pytest.raises(ValueError):
            self.cart_grid.get_index_of([test[0]], [test[1]])

    def test_get_index_of_good(self):
        test = (0.05, 0.15)
        test_idx = self.cart_grid.get_index_of([test[0]], [test[1]])
        numpy.testing.assert_allclose(test_idx, 0)

class TestCatalogBinning(unittest.TestCase):

    def setUp(self):
        # create some arbitrary grid
        self.nx = 8
        self.ny = 10
        self.dh = 0.1
        x_points = numpy.arange(self.nx) * self.dh
        y_points = numpy.arange(self.ny) * self.dh
        self.origins = list(itertools.product(x_points, y_points))
        # grid is missing first and last block
        self.origins.pop(0)
        self.origins.pop(-1)
        self.num_nodes = len(self.origins)
        # this is kinda ugly, maybe we want to create this in a different way, class method?
        self.cart_grid = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(self.origins, self.dh)], self.dh)

        # define an arbitrary magnitude range
        self.magnitudes = numpy.arange(2.5,8.5,0.1)

    def test_bin_spatial_counts(self):
        """ this will test both good and bad points within the region.

        1) 2 inside the domain
        2) outside the bbox but not in domain
        3) completely outside the domain

        we will check that only 1 event is placed in the grid and ensure its location is correct.
        """

        lons = numpy.array([0.05, 0.05, 0.15, -0.5])
        lats = numpy.array([0.05, 0.15, 0.05, -0.5])

        test_result = _bin_catalog_spatial_counts(lons, lats,
                                    self.cart_grid.num_nodes,
                                    self.cart_grid.mask,
                                    self.cart_grid.idx_map,
                                    self.cart_grid.xs,
                                    self.cart_grid.ys)

        # we have tested 2 inside the domain
        self.assertEqual(numpy.sum(test_result), 2)

        # we know that (0.05, 0.15) corresponds to index 0 from the above test
        self.assertEqual(test_result[0], 1)
        self.assertEqual(test_result[9], 1)

    def test_bin_spatial_probability(self):
        """ this will test both good and bad points within the region. added a point to the lons and lats
        to ensure that multiple events are only being counted once

        1) 2 inside the domain
        2) outside the bbox but not in domain
        3) completely outside the domain

        we will check that only 1 event is placed in the grid and ensure its location is correct.
        """

        lons = numpy.array([0.05, 0.05, 0.15, 0.15, -0.5])
        lats = numpy.array([0.05, 0.15, 0.05, 0.05, -0.5])

        test_result = _bin_catalog_probability(lons, lats,
                                    self.cart_grid.num_nodes,
                                    self.cart_grid.mask,
                                    self.cart_grid.idx_map,
                                    self.cart_grid.xs,
                                    self.cart_grid.ys)

        # we have tested 2 inside the domain
        self.assertEqual(numpy.sum(test_result), 2)

        # we know that (0.05, 0.15) corresponds to index 0 from the above test
        self.assertEqual(test_result[0], 1)
        self.assertEqual(test_result[9], 1)

    def test_bin_spatial_magnitudes(self):
        """ this will test both good and bad points within the region. added a point to the lons and lats
        to ensure that multiple events are only being counted once

        1) 2 inside the domain
        2) outside the bbox but not in domain
        3) completely outside the domain

        we will check that only 1 event is placed in the grid and ensure its location is correct.
        """

        lons = numpy.array([0.05, 0.05, 0.15, 0.15, -0.5])
        lats = numpy.array([0.05, 0.15, 0.05, 0.05, -0.5])
        mags = numpy.array([2.55, 2.65, 2.55, 2.05, 2.0])

        # expected bins None, (0, 1), (9, 0), None, None
        test_result, _ = _bin_catalog_spatio_magnitude_counts(lons, lats, mags,
                                                self.cart_grid.num_nodes,
                                                self.cart_grid.mask,
                                                self.cart_grid.idx_map,
                                                self.cart_grid.xs,
                                                self.cart_grid.ys,
                                                self.magnitudes)

        # we have tested 2 inside the domain
        self.assertEqual(numpy.sum(test_result), 2)

        # we know that (0.05, 0.15) corresponds to index 0 from the above test
        self.assertEqual(test_result[0, 1], 1)
        self.assertEqual(test_result[9, 0], 1)

if __name__ == '__main__':
    unittest.main()
