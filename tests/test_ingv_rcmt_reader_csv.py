import unittest
import os.path
import numpy
from csep.utils import readers
from csep.core import catalogs


def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Testing', 'ingv_rcmt-catalogs')
    return data_dir

class TestReadCatalog(unittest.TestCase):


    def test_read_short_cat(self):

        datadir = get_datadir()
        file_ = 'ingv_rcmt-short.csv'
        catalog_tuples = readers.read_ingv_rcmt_csv(os.path.join(datadir,file_))
        self.assertEqual(len(catalog_tuples),10)

    def test_read_large_cat(self):

        datadir = get_datadir()
        file_ = 'ingv_rcmt-large.csv'
        catalog_tuples = readers.read_ingv_rcmt_csv(os.path.join(datadir,file_))
        self.assertEqual(len(catalog_tuples),2340)
        self.assertEqual(catalog_tuples[891][6],77.)

class TestCreateCatalogFromRead(unittest.TestCase):

    def test_create_catalog(self):
        dtype = numpy.dtype([('longitude', numpy.float32),
                             ('latitude', numpy.float32),
                             ('year', numpy.int32),
                             ('month', numpy.int32),
                             ('day', numpy.int32),
                             ('magnitude', numpy.float32),
                             ('depth', numpy.float32),
                             ('hour', numpy.int32),
                             ('minute', numpy.int32),
                             ('second', numpy.int32)])
        datadir = get_datadir()
        filepath = os.path.join(datadir,'ingv_rcmt-large.csv')
        data_list = readers.read_ingv_rcmt_csv(filepath)
        catalog_array = numpy.array(data_list, dtype=dtype)
        Catalog = catalogs.ZMAPCatalog(catalog=catalog_array,
                                       compute_stats=False)
        Catalog.update_catalog_stats()
        self.assertEqual(Catalog.get_bvalue()[0], 0.49344819206533963)


if __name__ == '__main__':

    unittest.main()