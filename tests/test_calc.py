import unittest
from csep.utils.calc import bin1d_vec

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

