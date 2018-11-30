import numpy
from csep.core.evaluations import *

class MockCatalog:
    """
    Mock catalog class for testing purposes.
    """
    def __init__(self, val, name='Mock Catalog'):
        self.val = val
        self.name = name

    def __str__(self):
        return ''

    def get_number_of_events(self):
        return self.val

class TestNTest:
    '''
    n-test returns two values, delta_1 and delta_2
    delta 1 is prob of at most N_obs events given distribution from forecast
    delta 2 is prob at least N_obs events given distribution from forecast
    '''
    def test_greater_equal_side(self):
        n_obs = 0
        # have a vector with 50 values = 0 and 50 values = 1
        ints=numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result, ax = number_test(sets, obs)

        assert numpy.isclose(result[0], 1.0)
        assert numpy.isclose(result[1], 0.5)

    def test_less_equal_side(self):
        n_obs = 1
        # have a vector with 50 values = 0 and 50 values = 1
        ints=numpy.zeros(100)
        ints[:50] = 1

        sets = [MockCatalog(val) for val in ints]
        obs = MockCatalog(n_obs, 'Mock Obs.')

        result, ax = number_test(sets, obs)

        # result = (delta_1, delta_2)... at least, at most
        assert numpy.isclose(result[0], 0.5)
        assert numpy.isclose(result[1], 1.0)
