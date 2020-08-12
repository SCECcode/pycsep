import unittest
import numpy
from csep.core.catalogs import CSEPCatalog


class TestCreateCatalog(unittest.TestCase):

    def test_catalog_create_with_numpy_array(self):
        test = CSEPCatalog(catalog=numpy.array([1,2,3,4,5,6]), compute_stats=False)
        self.assertTrue(isinstance(test.catalog, numpy.ndarray))

    def test_catalog_create_with_failure(self):
        catalog = "failure condition, because catalog can't be string!"
        with self.assertRaises(TypeError):
            CSEPCatalog(catalog=catalog)
