import unittest
import itertools
import pytest

import numpy

from csep.core.regions import (
    CartesianGrid2D,
    QuadtreeGrid2D,
    compute_vertex, compute_vertices,
    _bin_catalog_spatio_magnitude_counts,
    _bin_catalog_spatial_counts,
    _bin_catalog_probability,
    quadtree_grid_bounds,
    california_relm_region,
    geographical_area_from_bounds
)

from csep.models import Polygon


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
        self.magnitudes = numpy.array([4, 5])
        # this is kinda ugly, maybe we want to create this in a different way, class method?
        self.cart_grid = CartesianGrid2D([Polygon(bbox) for bbox in compute_vertices(self.origins, self.dh)], self.dh, magnitudes=self.magnitudes)

    def test_polygon_mask_all_masked(self):
        polygons = [Polygon(bbox) for bbox in compute_vertices(self.origins, self.dh)]
        n_poly = len(polygons)
        # this should mask every polygon for easy checking
        test_grid = CartesianGrid2D(polygons, self.dh, mask=numpy.zeros(n_poly))
        self.assertEqual(n_poly, test_grid.num_nodes)
        numpy.testing.assert_array_equal(test_grid.bbox_mask, 1)
        numpy.testing.assert_array_equal(test_grid.poly_mask, 0)

    def test_object_creation(self):
        self.assertEqual(self.cart_grid.dh, self.dh, 'dh did not get initialized properly')
        self.assertEqual(self.cart_grid.num_nodes, self.num_nodes, 'num nodes is not correct')

    def test_xs_and_xy_correct(self):

        test_xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        test_ys = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        numpy.testing.assert_array_equal(self.cart_grid.xs, test_xs)
        numpy.testing.assert_array_equal(self.cart_grid.ys, test_ys)

    def test_bitmask_indices_mapping(self):
        test_idx = self.cart_grid.idx_map[1,0]
        numpy.testing.assert_array_equal(test_idx, 0, err_msg='mapping for first polygon index (good) not correct')

        test_idx = self.cart_grid.idx_map[0,1]
        numpy.testing.assert_array_equal(test_idx, 9, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[2,0]
        numpy.testing.assert_array_equal(test_idx, 1, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[0,2]
        numpy.testing.assert_array_equal(test_idx, 19, err_msg='mapping for polygon (good) not correct.')

        test_idx = self.cart_grid.idx_map[-1,-1]
        numpy.testing.assert_array_equal(test_idx, numpy.nan, err_msg='mapping for last index (bad) not correct.')

        test_idx = self.cart_grid.idx_map[0,0]
        numpy.testing.assert_array_equal(test_idx, numpy.nan, err_msg='mapping for first index (bad) not correct.')

    def test_domain_mask(self):
        test_flag = self.cart_grid.bbox_mask[0, 0]
        self.assertEqual(test_flag, 1)

        test_flag = self.cart_grid.bbox_mask[-1, 1]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.bbox_mask[1, -1]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.bbox_mask[2, 2]
        self.assertEqual(test_flag, 0)

        test_flag = self.cart_grid.bbox_mask[-1, -1]
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

    def test_to_from_dict(self):
        self.assertEqual(self.cart_grid, CartesianGrid2D.from_dict(self.cart_grid.to_dict()))

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
                                                  self.cart_grid.bbox_mask,
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
                                               self.cart_grid.bbox_mask,
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
                                                              self.cart_grid.bbox_mask,
                                                              self.cart_grid.idx_map,
                                                              self.cart_grid.xs,
                                                              self.cart_grid.ys,
                                                              self.magnitudes)

        # we have tested 2 inside the domain
        self.assertEqual(numpy.sum(test_result), 2)

        # we know that (0.05, 0.15) corresponds to index 0 from the above test
        self.assertEqual(test_result[0, 1], 1)
        self.assertEqual(test_result[9, 0], 1)

    def test_binning_using_bounds(self):
        """ Tests whether point correctly fall within the bin edges.

        This test is not exhaustive but checks a few points within the region and verifies that the points correctly fall
        within the bin edges.
        """
        gr = california_relm_region()

        # test points
        lons = numpy.array([-117.430, -117.505, -117.466, -117.5808, -117.612])
        lats = numpy.array([35.616, 35.714, 35.652, 35.7776, 35.778])

        # directly compute the indexes from the region object
        idxs = gr.get_index_of(lons, lats)
        for i, idx in enumerate(idxs):
            found_poly = gr.polygons[idx]
            lon = lons[i]
            lat = lats[i]

            assert lon >= found_poly.points[1][0] and lon < found_poly.points[2][0]
            assert lat >= found_poly.points[0][1] and lat < found_poly.points[2][1]


class TestQuadtreeGrid2D(unittest.TestCase):

    def setUp(self):
        self.zoom = 5
        self.mbins = numpy.arange(5.95, 8.95, 0.1)
        self.grid = QuadtreeGrid2D.from_single_resolution(self.zoom, magnitudes=self.mbins)

    def test_get_index(self):
        lons = [0, 45, 60, -180]
        lats = [0, 45, 60, -85.05]
        idx = numpy.array([426, 410, 403, 682])
        numpy.testing.assert_array_equal(self.grid.get_index_of(lons, lats), idx)

        # point outside
        numpy.testing.assert_array_equal(self.grid.get_index_of(0, 85.6), numpy.array([]))

    def test_quadtree_bounds(self):
        qk = ['0', '1']
        bounds = [[-180., 0., 0., 85.0511287798066], [0., 0., 180.,85.0511287798066]]
        numpy.testing.assert_array_equal(quadtree_grid_bounds(qk), bounds)

    def test_wrong_coordinates(self):
        lons = [180, -180]
        lats = [-85.06, 85.06]  # Lats outside the quadtree grid
        idx = []
        numpy.testing.assert_array_equal(self.grid.get_index_of(lons, lats), idx)

    def test_corner_points(self):
        # (lon, lat) = (0,0) lies on the top right corner of quadtree cell '21111'.
         # But it should belong to the top-right diagonal cell, i.e. '12222'.
         lon1 = 0
         lat1 = 0
         qk_cell1 = '12222'

         # Anything little less than (0,0) goes into the lower-left diagonal quadtree cell '21111'
         lon2 = -0.0000000001
         lat2 = -0.0000000001
         qk_cell2 = '21111'
         numpy.testing.assert_array_equal(self.grid.quadkeys[self.grid.get_index_of(lon1, lat1)], qk_cell1)
         numpy.testing.assert_array_equal(self.grid.quadkeys[self.grid.get_index_of(lon2, lat2)], qk_cell2)

    def test_num_cells(self):
         total_cells = 1024
         self.assertEqual(total_cells, self.grid.num_nodes)

    def test_find_quadkey_of_coord(self):
        lon = 0
        lat = 0
        qk_cell = '12222'
        numpy.testing.assert_array_equal(self.grid.quadkeys[self.grid.get_index_of(lon, lat)], qk_cell)


def test_geographical_area_from_bounds():
    area_globe = 510064471.90978825
    area_equator = 12363.6839902611
    numpy.testing.assert_array_equal(geographical_area_from_bounds(-180,-90, 180, 90), area_globe)
    numpy.testing.assert_array_equal(geographical_area_from_bounds(0,0,1,1), area_equator)


if __name__ == '__main__':
    unittest.main()
