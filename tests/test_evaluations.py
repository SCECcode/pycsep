import numpy
import unittest
from csep.core.evaluations import *
from csep.core.evaluations import _simulate_catalog


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


class TestNTest:
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

        # calculated by hand given the expected catalog
        self.assertEqual(simulated_ll[0], -7.178053830347945)