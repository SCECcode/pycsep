import numpy
import unittest
from csep.utils.stats import *


class TestEcdf:
    def test_ecdf_simple(self):
        """
        should have property that f(x) = P(X ≤ x).

        f(x) = count where t ≤ x / total number in x
        """
        test_case = unittest.TestCase()
        test_data = numpy.array([2,1,4,5])
        test_x, test_y = ecdf(test_data)
        test_case.assertListEqual(test_x.tolist(),[1,2,4,5])
        test_case.assertListEqual(test_y.tolist(),[0.25,0.5,0.75,1.0])

    def test_ecdf_multiple(self):
        test_case = unittest.TestCase()
        test_data = numpy.array([0,0,0,1])
        tx, ty = ecdf(test_data)
        test_case.assertListEqual(tx.tolist(),[0,0,0,1])
        test_case.assertListEqual(ty.tolist(),[0.25,0.5,0.75,1.0])

class TestGreaterEqualEcdf:
    def test_lower_bound(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 0.25
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 1.0

    def test_upper_bound(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 5.5
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 0.0

    def test_largest(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 5
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 0.25

    def test_smallest(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 1
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 1.0

    def test_repeating(self):
        test_data = numpy.array([0,0,0,0])
        test_val = 0
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 1.0

    def test_between(self):
        test_data = numpy.array([1,2,3,4,5])
        test_val = 1.5
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 0.8

    def test_non_integer(self):
        test_data = numpy.array([0.1,0.2,0.4,0.5])
        test_val = 0.2
        test_result = greater_equal_ecdf(test_data, test_val)
        assert test_result == 0.75

class TestLessEqualEcdf:
    def test_lower_bound(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 0.25
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 0.0

    def test_upper_bound(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 5.5
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 1.0

    def test_largest(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 5.0
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 1.0

    def test_smallest(self):
        test_data = numpy.array([1,2,4,5])
        test_val = 1
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 0.25

    def test_repeating(self):
        test_data = numpy.array([0,0,0,0.1])
        test_val = 0
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 0.75

    def test_between(self):
        test_data = numpy.array([1,2,3,4,5])
        test_val = 1.5
        test_result = less_equal_ecdf(test_data, test_val)
        # 1 out of 5 in test_data are <= than test_value
        assert test_result == 0.2

    def test_non_integer(self):
        test_data = numpy.array([0.1,0.2,0.4,0.5])
        test_val = 0.2
        test_result = less_equal_ecdf(test_data, test_val)
        assert test_result == 0.50
