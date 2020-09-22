import unittest
import os.path

import csep
from csep.utils import readers


def get_horus_path():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'ingv_catalogs', 
                            'HORUS_Ita_Catalog.txt')
    return data_dir

def get_emrcmt_path():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'ingv_catalogs',
                            'EuroMedCentrMomTensors.csv')
    return data_dir

class TestReadCatalog(unittest.TestCase):

    def test_cat_emrcmt(self):
        filepath = get_emrcmt_path()
        catalog_tuples = readers.ingv_emrcmt(filepath)
    
        #Test removal of wrong mw events and repeated events
        self.assertEqual(len(catalog_tuples),3)   
        #Test the fixing of date and time typos (epoch obtained manually)
        self.assertEqual(catalog_tuples[0][1],1547515505000)
        self.assertEqual(catalog_tuples[2][1],1576119900000)
        
        #Test reader
        target_lat = []
        target_lon = []
        target_depth = []
        target_mw = []
        with open(filepath) as file_:
            for targetline in file_.readlines():
                targetline_ = targetline.split(',')
                target_lon.append(float(targetline_[5].replace('"', '')))
                target_lat.append(float(targetline_[4].replace('"', '')))
                target_depth.append(float(targetline_[6].replace('"', '')))
                target_mw.append(float(targetline_[-3].replace('"', '')))

        for event in catalog_tuples:
            self.assertIn(event[2], target_lat)
            self.assertIn(event[3], target_lon)
            self.assertIn(event[4], target_depth)
            self.assertIn(event[5], target_mw)

    def test_cat_horus(self):
        filepath = get_horus_path()
        catalog_tuples = readers.ingv_horus(filepath)
        
        self.assertEqual(len(catalog_tuples), 2)
        #Test the fixing of over times typos (e.g. 61 sec) 
        self.assertEqual(catalog_tuples[1][1], 1600772464000)


if __name__ == '__main__':
    unittest.main()
