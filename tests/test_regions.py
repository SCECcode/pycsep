import os
import unittest

import numpy

from csep.core.regions import italy_csep_region, california_relm_region, nz_csep_region

def get_italy_region_fname():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'regions', 'ItalyTestArea.dat')
    return data_dir

def get_california_region_fname():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'regions', 'RELMTestArea.dat')
    return data_dir

def get_nz_region_fname():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'regions', 'NZTestArea.dat')
    return data_dir

class TestItalyRegion(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):

        self.from_dat = numpy.loadtxt(get_italy_region_fname())
        self.num_nodes = len(self.from_dat)

    def test_node_count(self):
        """ Ensures the node counts are consistent between the two files. """
        r = italy_csep_region()
        self.assertEqual(self.num_nodes, r.num_nodes)


    def test_origins(self):
        """ Compares XML file against the simple .dat file containing the region. """
        r = italy_csep_region()
        # they dont have to be in the same order, but they need
        numpy.testing.assert_array_equal(r.midpoints().sort(), self.from_dat.sort())

class TestCaliforniaRelmRegion(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.r = california_relm_region()
        self.from_dat = numpy.loadtxt(get_california_region_fname())
        self.num_nodes = len(self.from_dat)

    def test_node_count(self):
        """ Ensures the node counts are consistent between the two files. """
        self.assertEqual(self.num_nodes, self.r.num_nodes)

    def test_origins(self):
        """ Compares XML file against the simple .dat file containing the region. """
        # they dont have to be in the same order, but they need
        numpy.testing.assert_array_equal(self.r.midpoints().sort(), self.from_dat.sort())

    def test_eq_oper(self):
        r = california_relm_region()
        assert self.r == r

class TestNZRegion(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):

        self.from_dat = numpy.loadtxt(get_nz_region_fname())
        self.num_nodes = len(self.from_dat)

    def test_node_count(self):
        """ Ensures the node counts are consistent between the two files. """
        r = nz_csep_region()
        self.assertEqual(self.num_nodes, r.num_nodes)


    def test_origins(self):
        """ Compares XML file against the simple .dat file containing the region. """
        r = nz_csep_region()
        # they dont have to be in the same order, but they need
        numpy.testing.assert_array_equal(r.midpoints().sort(), self.from_dat.sort())
