import os
import unittest
import datetime
import xml.etree.ElementTree as ET

from csep.core.evaluations import *
from csep.core.evaluations import _simulate_catalog
from csep.core.catalogs import ZMAPCatalog
from csep.core.forecasts import GriddedForecast
from csep.utils.readers import read_csep1_zmap_ascii


def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Testing')
    return data_dir

class MockCatalog:
    """
    Mock catalog class for testing purposes.
    """
    def __init__(self, val, name='Mock Catalog'):
        self.val = val
        self.name = name
        self.event_count = self.get_number_of_events()

    def __str__(self):
        return ''

    def get_number_of_events(self):
        return self.val


class TestCSEP1NTestThreeMonthsEEPAS(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = get_datadir()
        self.forecast_start_date = datetime.datetime(2007,12,1,0,0,0)
        self.forecast_end_date = datetime.datetime(2008,3,1,0,0,0)
        self.test_date = datetime.datetime(2007,12,10,0,0,0)
        self.forecast_fname = os.path.join(self.root_dir,
                    'example_csep1_forecasts/Forecast/EEPAS-0F_12_1_2007.dat')
        self.catalog_fname = os.path.join(self.root_dir,
                    'example_csep1_forecasts/Observations/ThreeMonthsModel.catalog.nodecl.dat')
        self.result_fname = os.path.join(self.root_dir,
                'example_csep1_forecasts/Evaluations/NTest/NTest_Result/rTest_N-Test_EEPAS-0F_12_1_2007.xml')

    def test_ntest_three_months_eepas_model(self):
        """Tests N-Test implementation using EEPAS forecasts defined during CSEP1 testing phase.

        This test shows a guideline on how other tests from CSEP1 can be ported to test CSEP2 code. Test will
        read in a forecast in CSEP1 .dat format along with an EvaluationResult in XML format and verify the results.
        The forecast is a Three-Month EEPAS forecast dated 12-1-2007.dat and evaluated at 12-10-2007.
        """
        test_evaluation_dict = {}
        # load the forecast file
        print(self.forecast_fname)
        fore = GriddedForecast.from_csep1_ascii(self.forecast_fname, self.forecast_start_date, self.forecast_end_date)
        # scale the forecast to the test_date, assuming the forecast is rate over the period specified by forecast start and end dates
        fore.scale_to_test_date(self.test_date)
        # load the catalog file
        cata = ZMAPCatalog(catalog=read_csep1_zmap_ascii(self.catalog_fname), region=fore)
        # compute n_test_result
        res = csep1_number_test(fore, cata)
        # parse xml result from file
        result_dict = self._parse_xml_result()
        # build evaluation_dict
        test_evaluation_dict['delta1'] = res.quantile[0]
        test_evaluation_dict['delta2'] = res.quantile[1]
        test_evaluation_dict['event_count_forecast'] = fore.event_count
        test_evaluation_dict['event_count'] = cata.event_count
        # comparing floats, so we will just map to ndarray and use allclose
        expected = numpy.array([v for k,v in result_dict.items()])
        computed = numpy.array([v for k,v in test_evaluation_dict.items()])
        print(expected)
        print(computed)
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

class TestCatalogBasedNTest:
    '''
    n-test returns two values, delta_1 and delta_2
    delta 1 is prob of at least N_obs events given distribution from forecast
    delta 2 is prob at most N_obs events given distribution from forecast
    '''
    def test_greater_equal_side(self):
        n_obs = 0
        # have a vector with 50 values = 0 and 50 values = 1
        ints=numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result = number_test(sets, obs)

        assert numpy.isclose(result.quantile[0], 1.0)
        assert numpy.isclose(result.quantile[1], 0.5)

    def test_less_equal_side(self):
        n_obs = 1
        # have a vector with 50 values = 0 and 50 values = 1
        ints = numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result = number_test(sets, obs)

        # result = (delta_1, delta_2)... at least, at most
        assert numpy.isclose(result.quantile[0], 0.5)
        assert numpy.isclose(result.quantile[1], 1.0)

    def test_obs_greater_than_all(self):
        n_obs = 2
        # have a vector with 50 values = 0 and 50 values = 1
        ints = numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result = number_test(sets, obs)

        # result = (delta_1, delta_2)... at least, at most
        assert numpy.isclose(result.quantile[0], 0.0)
        assert numpy.isclose(result.quantile[1], 1.0)

    def test_obs_less_than_all(self):
        n_obs = -1
        # have a vector with 50 values = 0 and 50 values = 1
        ints = numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result = number_test(sets, obs)

        # result = (delta_1, delta_2)... at least, at most
        assert numpy.isclose(result.quantile[0], 1.0)
        assert numpy.isclose(result.quantile[1], 0.0)


class TestPoissonLikelihood(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        numpy.random.seed(self.seed)
        # used for posterity
        self.random_matrix = numpy.random.rand(2, 2)
        self.forecast_data = numpy.array([[1, 1], [1, 1]])
        self.observed_data = numpy.array([[1, 1], [1, 1]])

    def test_simulate_catalog(self):
        # expecting the sampling weights to be [0.25, 0.5, 0.75, 1.0]
        # assuming the random numbers are equal to thhe following:
        random_numbers = numpy.array([[0.5488135, 0.71518937],
                                      [0.60276338, 0.54488318]])

        num_events = 4

        # ensures that our random numbers used to manually create expected catalog are consistent
        numpy.testing.assert_allclose(random_numbers, self.random_matrix)

        # assuming that our forecast_data are uniform as defined above
        # bin[0] = [0, 0.25)
        # bin[1] = [0.25, 0.5)
        # bin[2] = [0.5, 0.75)
        # bin[3] = [0.75, 1.0)
        expected_catalog = [0, 0, 4, 0]

        # forecast_data should be event counts
        expected_forecast_count = numpy.sum(self.forecast_data)

        # used to determine where simulated earthquake shoudl be placed
        sampling_weights = numpy.cumsum(self.forecast_data.ravel()) / expected_forecast_count

        # this is taken from the test likelihood function
        sort_args = numpy.argsort(sampling_weights)
        sim_fore = numpy.empty(sampling_weights.shape)
        sim_fore = _simulate_catalog(num_events, sampling_weights, sort_args, sim_fore,
                                     random_numbers=self.random_matrix)

        # final statement
        numpy.testing.assert_allclose(expected_catalog, sim_fore)

    def test_likelihood(self):
        qs, obs_ll, simulated_ll = poisson_likelihood_test(self.forecast_data, self.observed_data, num_simulations=1,
                                                           seed=0)

        # very basic result to pass "laugh" test
        self.assertEqual(qs, 1)

        # forecast and observation are the same, sum(np.log(poisson(1, 1))) = -4
        self.assertEqual(obs_ll, -4)

        # calculated by hand given the expected catalog, see explanation in zechar et al., 2010.
        self.assertEqual(simulated_ll[0], -7.178053830347945)

