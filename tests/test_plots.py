import numpy
import matplotlib.pyplot
from unittest import mock
import unittest
import random
import string
from csep.utils import plots


class TestPoissonPlots(unittest.TestCase):

    def test_SingleNTestPlot(self):

        expected_val = numpy.random.randint(0,20)
        observed_val = numpy.random.randint(0, 20)
        Ntest_result = mock.Mock()
        Ntest_result.name = 'Mock NTest'
        Ntest_result.sim_name = 'Mock SimName'
        Ntest_result.test_distribution = ['poisson', expected_val]
        Ntest_result.observed_statistic = observed_val
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntest_result)

        self.assertEqual(matplotlib.pyplot.gca().collections, ax.collections)
        self.assertEqual([i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                    [i.sim_name for i in [Ntest_result]])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntest_result.name)

    def test_MultiNTestPlot(self,show=False):

        n_plots = numpy.random.randint(1,20)
        Ntests = []
        for n in range(n_plots):
            Ntest_result = mock.Mock()
            Ntest_result.name = 'Mock NTest'
            Ntest_result.sim_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
            Ntest_result.test_distribution = ['poisson', numpy.random.randint(0, 20)]
            Ntest_result.observed_statistic = numpy.random.randint(0, 20)
            Ntests.append(Ntest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntests)
        Ntests.reverse()

        self.assertEqual(matplotlib.pyplot.gca().collections, ax.collections)
        self.assertEqual([i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                        [i.sim_name for i in Ntests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntests[0].name)
        if show:
            matplotlib.pyplot.show()

    def test_MultiSTestPlot(self,show=False):

        s_plots = numpy.random.randint(1,20)
        Stests = []
        for n in range(s_plots):
            Stest_result = mock.Mock() # Mock class with random attributes
            Stest_result.name = 'Mock STest'
            Stest_result.sim_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
            Stest_result.test_distribution = numpy.random.uniform(-1000, 0, numpy.random.randint(3, 500)).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(-1000, 0) #random observed statistic
            if numpy.random.random() <0.02: # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Stests)
        Stests.reverse()

        self.assertEqual(matplotlib.pyplot.gca().collections, ax.collections)
        self.assertEqual([i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                        [i.sim_name for i in Stests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Stests[0].name)


if __name__ == '__main__':
    unittest.main()

