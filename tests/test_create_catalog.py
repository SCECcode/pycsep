# Python imports
import unittest
import tempfile
import os

# Third-party imports
import numpy

# PyCSEP imports
import csep
from csep.utils.datasets import comcat_example_catalog_fname
from csep.core.catalogs import CSEPCatalog



class TestCreateCatalog(unittest.TestCase):

    def test_catalog_create_with_numpy_array(self):
        test = CSEPCatalog(data=numpy.array([1,2,3,4,5,6]), compute_stats=False)
        self.assertTrue(isinstance(test.catalog, numpy.ndarray))

    def test_catalog_create_with_failure(self):
        catalog = "failure condition, because catalog can't be string!"
        with self.assertRaises(TypeError):
            CSEPCatalog(data=catalog)

class TestCatalogSerialization(unittest.TestCase):

    def setUp(self):

        self.test_catalog = csep.load_catalog(comcat_example_catalog_fname)

    def test_dict(self):
        """ Tests to_dict() and from_dict() methods.

        This serialization is lossless because all information about the catalogs are preserved and original class
        objects are recovered. This test will ensure this occurs properly.
        """

        catalog_dict = self.test_catalog.to_dict()
        catalog_from_dict = CSEPCatalog.from_dict(catalog_dict)

        self.assertTrue(self.test_catalog == catalog_from_dict)


    def test_dataframe(self):
        """ Tests to_dataframe() and from_dataframe() methods.

        These are lossy methods because only the catalog data are preserved. The catalog statistics should still
        be the same but the date_accessed, filters, and region information will be lost during this exchange.
        """

        catalog_df = self.test_catalog.to_dataframe(with_datetime=True)
        catalog_from_df = CSEPCatalog.from_dataframe(catalog_df)

        # lossy serialization, thus need to bind date accessed
        catalog_from_df.date_accessed = self.test_catalog.date_accessed

        self.assertTrue(self.test_catalog == catalog_from_df)


    def test_json(self):
        """ Tests writing catalog class to and from json files.

        JSON writing piggy-backs on the to_dict() and from_dict() methods, but has the extra layer of encoding to JSON
        representation. This test will write temporary files.
        """

        with tempfile.TemporaryDirectory() as tempdir:
            json_fname = os.path.join(tempdir, 'test_catalog.json')
            self.test_catalog.write_json(json_fname)
            catalog_from_json = CSEPCatalog.load_json(json_fname)
        self.assertTrue(self.test_catalog == catalog_from_json)


    def test_ascii(self):
        """ Tests writing catalogs to and from ascii files.

        The catalog objects created need their date_accessed attributes synchronized, because this attribute is not
        included in the ascii catalog description.
        """

        with tempfile.TemporaryDirectory() as tempdir:
            ascii_fname = os.path.join(tempdir, 'test_catalog.csv')
            self.test_catalog.write_ascii(ascii_fname)
            catalog_from_ascii = CSEPCatalog.load_catalog(ascii_fname)

        # need to synchronize for testing
        catalog_from_ascii.date_accessed = self.test_catalog.date_accessed
        self.assertTrue(self.test_catalog == catalog_from_ascii)

