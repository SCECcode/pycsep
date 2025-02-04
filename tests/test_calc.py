import unittest
import numpy
from csep.utils.calc import bin1d_vec, cleaner_range


class TestCleanerRange(unittest.TestCase):

    def setUp(self):

        self.start = 0.0
        self.end = 0.9
        self.dh = 0.1
        self.truth = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_discrepancy_with_arange_catch_failure(self):

        ar = numpy.arange(self.start, self.end + self.dh / 2, self.dh)
        cr = cleaner_range(self.start, self.end, self.dh)

        self.assertRaises(AssertionError, numpy.testing.assert_array_equal, ar, cr)
        self.assertRaises(AssertionError, numpy.testing.assert_array_equal, ar, self.truth)


    def test_discrepancy_with_direct_input(self):

        cr = cleaner_range(self.start, self.end, self.dh)
        numpy.testing.assert_array_equal(self.truth, cr)

class TestBin1d(unittest.TestCase):

    def test_bin1d_vec(self):
        data = [0.34, 0.35]
        bin_edges = [0.33, 0.34, 0.35, 0.36]
        test = bin1d_vec(data, bin_edges).tolist()
        expected = [1, 2]
        self.assertListEqual(test, expected)

    def test_bin1d_vec2(self):
        data = [0.9999999]
        bin_edges = [0.8, 0.9, 1.0]
        test = bin1d_vec(data, bin_edges)
        expected = [1]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec3(self):
        data = [-118.9999999]
        bin_edges = [-119.0, -118.9, -118.8]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec4(self):
        data = [-118.9999999]
        bin_edges = [-119.0, -118.98, -118.96]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec5(self):
        data = [-119.0]
        bin_edges = [-119.0, -118.98, -118.96]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec6(self):
        data = [1189999.99999]
        bin_edges = [1189999.9, 1190000.0, 1200000.0]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec7(self):
        data = [-118.98]
        bin_edges = [-119.0, -118.98, -118.96]
        test = bin1d_vec(data, bin_edges)
        expected = [1]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec8(self):
        data = [-118.9600000001]
        bin_edges = [-119.0, -118.98, -118.96]
        test = bin1d_vec(data, bin_edges)
        expected = [1]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec9(self):
        data = [-118.97999999]
        bin_edges = [-119.0, -118.98, -118.96]
        test = bin1d_vec(data, bin_edges)
        expected = [1]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_vec_int(self):
        data = [1, 3, 5, 10, 20]
        bin_edges = [0, 10, 20, 30]
        test = bin1d_vec(data, bin_edges)
        expected = [0, 0, 0, 1, 2]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_single_bin1(self):
        data = [-1, 0, 2, 3, 1, 1.5, 1.0, 0.999999999999999]
        bin_edges = [1]
        # purposely leaving right_continous flag=False bc it should be forced in the bin1d_vec function
        test = bin1d_vec(data, bin_edges)
        expected = [-1, -1, 0, 0, 0, 0, 0, -1]
        self.assertListEqual(test.tolist(), expected)

    def test_bin1d_single_bin2(self):
        data = [4.0]
        bin_edges = [3.0]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_upper_limit_right_continuous(self):
        data = [40, 40, 40]
        bin_edges = [0, 10, 20, 30]
        test = bin1d_vec(data, bin_edges, right_continuous=True)
        expected = [3, 3, 3]
        self.assertListEqual(test.tolist(), expected)

    def test_upper_limit_not_continuous(self):
        data = [30, 30, 30]
        bin_edges = [0, 10, 20, 30]
        test = bin1d_vec(data, bin_edges)
        expected = [3, 3, 3]
        self.assertListEqual(test.tolist(), expected)

    def test_lower_limit(self):
        data = [0]
        bin_edges = [0, 10, 20, 30]
        test = bin1d_vec(data, bin_edges)
        expected = [0]
        self.assertListEqual(test.tolist(), expected)

    def test_less_and_greater_than(self):
        data = [-1, 35, 40]
        bin_edges = [0, 10, 20, 30]
        test = bin1d_vec(data, bin_edges)
        expected = [-1, 3, -1]
        self.assertListEqual(test.tolist(), expected)

    def test_scalar_inside(self):
        mbins = numpy.arange(5.95, 9, 0.1)  # (magnitude) bins from 5.95 to 8.95

        for i, m in enumerate(mbins):
            idx = bin1d_vec(m, mbins, right_continuous=True)  # corner cases
            self.assertEqual(idx, i)

            idx = bin1d_vec(m + 0.05, mbins, right_continuous=True)  # center bins
            self.assertEqual(idx, i)

            idx = bin1d_vec(m + 0.099999999, mbins, right_continuous=True)  # end of bins
            self.assertEqual(idx, i)

        idx = bin1d_vec(10, mbins, right_continuous=True)  # larger than last bin edge
        self.assertEqual(idx, mbins.size - 1)

    def test_scalar_outside(self):
        mbins = numpy.arange(5.95, 9, 0.1)  # (magnitude) bins from 5.95 to 8.95

        idx = bin1d_vec(5, mbins, right_continuous=True)
        self.assertEqual(idx, -1)

        idx = bin1d_vec(4, mbins, right_continuous=True)
        self.assertEqual(idx, -1)
