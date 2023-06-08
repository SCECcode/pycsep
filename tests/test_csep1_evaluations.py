#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:38:30 2020
Implementation of Unit Test for L Test
@author: khawaja
"""
import os
import numpy
import unittest
import xml.etree.ElementTree as ET
import datetime

from csep.core.poisson_evaluations import _number_test_ndarray, _w_test_ndarray, _t_test_ndarray, number_test
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog
from csep.utils.readers import zmap_ascii

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'example_csep1_forecasts')
    return data_dir


class TestCSEP1NTestThreeMonthsEEPAS(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = get_datadir()
        self.forecast_start_date = datetime.datetime(2007,12,1,0,0,0)
        self.forecast_end_date = datetime.datetime(2008,3,1,0,0,0)
        self.test_date = datetime.datetime(2007,12,10,0,0,0)
        self.forecast_fname = os.path.join(self.root_dir,
                    'Forecast/EEPAS-0F_12_1_2007.dat')
        self.catalog_fname = os.path.join(self.root_dir,
                    'Observations/ThreeMonthsModel.catalog.nodecl.dat')
        self.result_fname = os.path.join(self.root_dir,
                'Evaluations/NTest/NTest_Result/rTest_N-Test_EEPAS-0F_12_1_2007.xml')

    def test_ntest_three_months_eepas_model(self):
        """Tests N-Test implementation using EEPAS forecasts defined during CSEP1 testing phase.

        This test shows a guideline on how other tests from CSEP1 can be ported to test CSEP2 code. Test will
        read in a forecast in CSEP1 .dat format along with an EvaluationResult in XML format and verify the results.
        The forecast is a Three-Month EEPAS forecast dated 12-1-2007.dat and evaluated at 12-10-2007.
        """
        test_evaluation_dict = {}
        # load the forecast file
        print(self.forecast_fname)
        fore = GriddedForecast.load_ascii(self.forecast_fname, self.forecast_start_date, self.forecast_end_date)
        # scale the forecast to the test_date, assuming the forecast is rate over the period specified by forecast start and end dates
        fore.scale_to_test_date(self.test_date)
        # load the observed_catalog file
        cata = CSEPCatalog(data=zmap_ascii(self.catalog_fname), region=fore)
        # compute n_test_result
        res = number_test(fore, cata)
        # parse xml result from file
        result_dict = self._parse_xml_result()
        # build evaluation_dict
        test_evaluation_dict['delta1'] = res.quantile[0]
        test_evaluation_dict['delta2'] = res.quantile[1]
        test_evaluation_dict['event_count_forecast'] = fore.event_count
        test_evaluation_dict['event_count'] = cata.event_count
        # comparing floats, so we will just map to ndarray and use allclose
        expected = numpy.array([v for k, v in result_dict.items()])
        computed = numpy.array([v for k, v in test_evaluation_dict.items()])
        numpy.testing.assert_allclose(expected, computed, rtol=1e-5)


    def _parse_xml_result(self):
        # unfortunately these have to be different for each result file, because the XML schema changes
        xml_result = {}
        xml_abs_path = os.path.join(self.root_dir, self.result_fname)
        ns = {'ns0': "http://www.scec.org/xml-ns/csep/0.1"}
        tree = ET.parse(xml_abs_path)
        root = tree.getroot()
        result_elem = root.find('ns0:resultData', ns)
        # should only be a single child here
        for child in result_elem:
            xml_result['delta1'] = float(child.find('ns0:delta1', ns).text)
            xml_result['delta2'] = float(child.find('ns0:delta2', ns).text)
            xml_result['event_count_forecast'] = float(child.find('ns0:eventCountForecast', ns).text)
            xml_result['event_count'] = float(child.find('ns0:eventCount', ns).text)
        return xml_result


class TestGriddedForecastTests(unittest.TestCase):

    def test_n_test(self):
        forecast = numpy.zeros((10, 10))+0.0015
        forecast = forecast / forecast.size
        observation = numpy.zeros((10, 10))
        expected_output = (1.0, 0.9985011244377109)
        print('N Test: Running Unit Test')
        numpy.testing.assert_allclose(_number_test_ndarray(forecast.sum(), observation.sum()), expected_output)

    def test_w_test(self):
        pass
        """
        This function tests w test.
        There are two outputs of w test. Z_statistic and the probability value.
        """
        x = numpy.array([35.5,   44.5,  39.8,  33.3,  51.4,  51.3,  30.5,  48.9,   42.1,   40.3, 46.8,   38.0, 40.1,
                         36.8,  39.3,  65.4, 42.6,  42.8,  59.8,   52.4, 26.2,   60.9,  45.6,  27.1,  47.3,  36.6,
                         55.6,  45.1,   52.2,   43.5])
        median = 45

        expected_output = {'z_statistic': -0.6684710290340584, 'probability': 0.5038329688781412}
        print('W Test: Running Unit Test')
        self.assertEqual(_w_test_ndarray(x, median), expected_output, 'Failed W Test')

    def test_t_test(self):
        pass
        """
        This function tests the T test
        It compares the T_statistic, Information Gain, its upper and lower limits. Values were computed manually following
        the equations from

        """
        forecast_a = numpy.array([[8, 2], [3, 5]])
        forecast_b = numpy.array([[6, 4], [2, 8]])
        obs = numpy.array([[5, 8], [4, 2]])

        t_test_expected = {'t_statistic': 1.5385261717159382,
                           't_critical': 2.10092204024096,
                           'information_gain': 0.08052612477654024,
                           'ig_lower': -0.029435677283374914,
                           'ig_upper': 0.19048792683645538}
        numpy.testing.assert_allclose(
            [v for k, v in _t_test_ndarray(forecast_a, forecast_b, numpy.sum(obs), forecast_a.sum(), forecast_b.sum()).items()],
            [v for k, v in t_test_expected.items()])


if __name__ == '__main__':
    unittest.main() 

