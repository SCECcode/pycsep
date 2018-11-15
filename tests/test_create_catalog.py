import unittest
import numpy
import pandas
from csep.core.catalogs import BaseCatalog


class TestCreateCatalog(unittest.TestCase):

    def test_catalog_create_with_numpy_array(self):
        catalog = numpy.array([1,2,3,4,5,6])
        self.test = BaseCatalog(catalog=catalog)
        self.assertTrue(isinstance(self.test.catalog, numpy.ndarray))

    def test_catalog_create_with_failure(self):
        catalog = 'failure condition!'
        with self.assertRaises(ValueError):
            BaseCatalog(catalog=catalog)
