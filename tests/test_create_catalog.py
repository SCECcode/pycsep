import unittest
import numpy
from csep.core.catalogs import AbstractBaseCatalog


class TestCreateCatalog(unittest.TestCase):

    def test_catalog_create_with_numpy_array(self):
        catalog = numpy.array([1,2,3,4,5,6])
        self.test = AbstractBaseCatalog(catalog=catalog, compute_stats=False)
        self.assertTrue(isinstance(self.test.data, numpy.ndarray))

    def test_catalog_create_with_failure(self):
        catalog = 'failure condition!'
        with self.assertRaises(ValueError):
            AbstractBaseCatalog(catalog=catalog)
