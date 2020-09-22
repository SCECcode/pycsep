import unittest
import pytest
from csep.utils.calc import *
from csep.core.exceptions import CSEPException


class TestNearestIndex:
    """
    Tests the function that given any value, t, should return the index in x
    that satisfies argmin(xi+1 < t ≤ xi).
    """
    def test_within_bound_but_not_in_array(self):
        """
        tests it floors properly.
        """
        test_data = numpy.array([1,2,4,5])
        test_val = 1.2
        # should return 0
        test_result = nearest_index(test_data, test_val)
        assert test_result == 0

    def test_properly_flooring(self):
        """
        tests if rounding or flooring.
        """
        test_data = numpy.array([1,2,4,5])
        test_val = 1.8
        # still should return 0
        test_result = nearest_index(test_data, test_val)
        assert test_result == 1

    def test_value_in_array(self):
        """
        tests it handles the simple case properly.
        """
        test_data = numpy.array([1,2,4,5])
        test_val = 2
        # should return 1
        test_result = nearest_index(test_data, test_val)
        assert test_result == 1

    def test_less_than_all_in_array(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 0
        # should return 0
        test_result = nearest_index(test_data, test_val)
        assert test_result == 0

    def test_greater_than_all_in_array(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 6
        # should return 3
        test_result = nearest_index(test_data, test_val)
        assert test_result == 3


class TestDigitize(unittest.TestCase):

    """
    edge cases:
        -
    """

    def setUp(self):
        self.bin_edges = [0, 1, 2, 3, 4]

    def test_lower_bound(self):
        data = [-0.1, 0.2, 1.2, 2.2, 3.2]
        with pytest.raises(CSEPException):
            discretize(data, self.bin_edges)

    def test_upper_bound(self):
        data = [-0.1, 0.2, 1.2, 2.2, 4.1]
        with pytest.raises(CSEPException):
            discretize(data, self.bin_edges)

    def test_value_on_lower_bin_edge(self):
        data = [0, 0.2, 1.2, 2.2, 3.2]
        test = discretize(data, self.bin_edges)
        expected = [0, 0, 1, 2, 3]
        self.assertListEqual(test.tolist(), expected)

    def test_value_on_upper_bin_edge(self):
        data = [0.2, 1.0, 0.9, 3.4, 3.9]
        test = discretize(data, self.bin_edges)
        expected = [0, 1, 0, 3, 3]
        self.assertListEqual(test.tolist(), expected)

    def test_upper_bound_equal_right_continuous(self):
        data = [4.0]
        test = discretize(data, self.bin_edges, right_continuous=True)
        expected = [4]
        self.assertListEqual(test.tolist(), expected)

    def test_upper_bound_equal(self):
        data = [5.0]
        with pytest.raises(CSEPException):
            discretize(data, self.bin_edges)

    def test_unsorted_bins(self):
        data = [4.0]
        with pytest.raises(ValueError):
            discretize(data, self.bin_edges[::-1])

    def test_empty_data(self):
        data = []
        test = discretize(data, self.bin_edges)
        expected = []
        self.assertListEqual(test.tolist(), expected)

    def test_empty_bin_edges(self):
        data = [4.0]
        with pytest.raises(ValueError):
            discretize(data, [])

    def test_noninteger_bins(self):
        data = [2.24, 2.64, 2.99, 3.00]
        bin_edges = [2, 2.25, 2.5, 2.75, 3.00, 3.25]
        test = discretize(data, bin_edges)
        expected = [2, 2.5, 2.75, 3.00]
        self.assertListEqual(test.tolist(), expected)