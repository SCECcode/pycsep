from unittest import TestCase
from csep.utils.basic_types import GriddedDataSet


class TestGriddedDataSet(TestCase):
    def test_add(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        # bins = [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
        # counts = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.assertListEqual(gridded.data, [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertListEqual(gridded.bins, [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35])

    def test_add_again_no_bounds_change(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        gridded.add(data)
        self.assertListEqual(gridded.data, [2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0])
        self.assertListEqual(gridded.bins, [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35])

    def test_add_again_upper_bounds_change(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        gridded.add([0.36])
        self.assertListEqual(gridded.data, [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        self.assertListEqual(gridded.bins, [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37])

    def test_add_again_lower_bounds_change(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        gridded.add([0.23])
        self.assertListEqual(gridded.data, [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertListEqual(gridded.bins,
                             [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35])

    def test_add_again_both_bounds_change(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        gridded.add([0.23, 0.36])
        self.assertListEqual(gridded.data, [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        self.assertListEqual(gridded.bins,
                             [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37])

    def test_add_empty_list(self):
        gridded = GriddedDataSet(dh=0.1)
        data = [0.25, 0.34, 0.24, 0.26]
        gridded.add(data)
        gridded.add([])
        self.assertListEqual(gridded.data, [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertListEqual(gridded.bins, [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35])





