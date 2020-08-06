import os
import numpy
import unittest

from csep.core.poisson_evaluations import _simulate_catalog, _poisson_likelihood_test

def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'Comcat')
    return data_dir

class TestPoissonLikelihood(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        numpy.random.seed(self.seed)
        # used for posterity
        self.random_matrix = numpy.random.rand(1, 4)
        self.forecast_data = numpy.array([[1, 1], [1, 1]])
        self.observed_data = numpy.array([[1, 1], [1, 1]])

    def test_simulate_catalog(self):
        # expecting the sampling weights to be [0.25, 0.5, 0.75, 1.0]
        # assuming the random numbers are equal to thhe following:
        random_numbers = numpy.array([[0.5488135, 0.71518937, 0.60276338, 0.54488318]])

        num_events = 4

        # ensures that our random numbers used to manually create expected observed_catalog are consistent
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
        sim_fore = numpy.empty(sampling_weights.shape)
        sim_fore = _simulate_catalog(num_events, sampling_weights, sim_fore,
                                     random_numbers=self.random_matrix)

        # final statement
        numpy.testing.assert_allclose(expected_catalog, sim_fore)

        # test again to ensure that fill works properply
        sim_fore = _simulate_catalog(num_events, sampling_weights, sim_fore,
                                     random_numbers=self.random_matrix)

        # final statement
        numpy.testing.assert_allclose(expected_catalog, sim_fore)

    def test_likelihood(self):
        qs, obs_ll, simulated_ll = _poisson_likelihood_test(self.forecast_data, self.observed_data, num_simulations=1,
                                                            random_numbers=self.random_matrix, use_observed_counts=True)

        # very basic result to pass "laugh" test
        numpy.testing.assert_allclose(qs, 1)

        # forecast and observation are the same, sum(np.log(poisson(1, 1))) = -4
        numpy.testing.assert_allclose(obs_ll, -4)

        # calculated by hand given the expected data, see explanation in zechar et al., 2010.
        numpy.testing.assert_allclose(simulated_ll[0], -7.178053830347945)

