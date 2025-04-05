import os
import unittest

import numpy

from csep.core.regions import (
    italy_csep_region, italy_csep_collection_region,
    california_relm_region, california_relm_collection_region,
    nz_csep_region, nz_csep_collection_region,
)

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

class TestBin2d(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestBin2d, cls).setUpClass()

        # (loading those is the bottleneck of this test case)
        cls.regions = [
            italy_csep_region(),
            italy_csep_collection_region(),
            california_relm_region(),
            california_relm_collection_region(),
            nz_csep_region(),
            nz_csep_collection_region(),
            # global_region()  # extreme slow-down (~2min loading + ~5min per loop + ~4s per vect)
        ]

    def test_bin2d_regions_origins(self):
        """every origin must be inside its own bin
        """

        for region in self.regions:
            origins = region.origins()
            self._test_bin2d_region_loop(region, origins)
            self._test_bin2d_region_vect(region, origins)
            self._test_bin2d_region_vect(region, origins.astype(numpy.float32))

    def test_bin2d_regions_midpoints(self):
        """every midpoint must be inside its own bin
        """

        for region in self.regions:
            midpoints = region.midpoints()
            self._test_bin2d_region_loop(region, midpoints)
            self._test_bin2d_region_vect(region, midpoints)
            self._test_bin2d_region_vect(region, midpoints.astype(numpy.float32))

    def test_bin2d_regions_endcorner(self):
        """every corner point (~opposite end of the origin) must be inside its own bin
        """

        for region in self.regions:
            frac = 0.9999999999
            endcorners = region.origins() + frac*region.dh
            self._test_bin2d_region_loop(region, endcorners)
            self._test_bin2d_region_vect(region, endcorners)
            frac = 0.999  # decrease frac for float32 due to its lower resolution
            endcorners = region.origins() + frac*region.dh
            self._test_bin2d_region_vect(region, endcorners.astype(numpy.float32))

    def _test_bin2d_region_loop(self, region, coords):
        """(slow) loop over origins; each time, calls bin1d_vec for lat & lon scalars
        """

        for i, origin in enumerate(coords):
            idx = region.get_index_of(
                origin[0],
                origin[1],
            )
            self.assertEqual(i, idx)

    def _test_bin2d_region_vect(self, region, coords):
        """call bin1d_vec once for all lat origins & all lon origins

        Besides, also tests if vectors with ndim=2 are consumed properly
        (returns 2nd-order/nested list ([[...]]).
        """

        lons, lats = numpy.split(coords.T, 2)  # both have ndim=2!
        test = region.get_index_of(lons, lats).tolist()  # nested list ([[...]])
        expected = [numpy.arange(len(region.origins())).tolist()]  # embed in another list
        self.assertListEqual(test, expected)
