import unittest
import os.path
from csep.utils import readers
from csep.utils.time_utils import epoch_time_to_utc_datetime


def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Testing', 'ingv_rcmt-catalogs')
    return data_dir


class TestReadCatalog(unittest.TestCase):

    def test_cats(self):
        cat_dir = get_datadir()
        filepaths = os.listdir(cat_dir)

        for filepath in filepaths:

            catalog_tuples = readers.read_ingv_rcmt_csv(os.path.join(cat_dir, filepath))
            lines = 0

            with open(os.path.join(cat_dir, filepath)) as file_:
                for targetline, testline in zip(file_.readlines(), catalog_tuples):
                    targetline_ = targetline.split(',')

                    # lon/lat
                    self.assertAlmostEqual(float(targetline_[5].replace('"', '')), testline[3])  # In some files, some line's values are under "" and some not.
                    self.assertAlmostEqual(float(targetline_[4].replace('"', '')), testline[2]) # Python csv handles that well within the reader function
                    # date
                    test_dt = epoch_time_to_utc_datetime(testline[1])
                    testdate = "%i-%02i-%02i" % (test_dt.year, test_dt.month, test_dt.day)
                    self.assertEqual(targetline_[1], testdate)
                    # time
                    testhour = "%02i:%02i:%02i" % (test_dt.hour, test_dt.minute, test_dt.second)
                    self.assertEqual(targetline_[2].replace(' ', '0'), testhour) # In some events, time are written like '12:12: 1' instead of '12:12:01'
                    # depth
                    self.assertAlmostEqual(float(targetline_[6].replace('"', '')), testline[4])
                    # mw
                    self.assertAlmostEqual(float(targetline_[-3]), testline[5])
                    lines += 1

            self.assertEqual(len(catalog_tuples), lines)


if __name__ == '__main__':

    unittest.main()
