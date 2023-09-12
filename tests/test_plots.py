import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot
from unittest import mock
import unittest
import random
import string
from csep.utils import plots


class TestPoissonPlots(unittest.TestCase):

    def test_SingleNTestPlot(self):

        expected_val = numpy.random.randint(0, 20)
        observed_val = numpy.random.randint(0, 20)
        Ntest_result = mock.Mock()
        Ntest_result.name = 'Mock NTest'
        Ntest_result.sim_name = 'Mock SimName'
        Ntest_result.test_distribution = ['poisson', expected_val]
        Ntest_result.observed_statistic = observed_val
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntest_result)

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in [Ntest_result]])
        self.assertEqual(matplotlib.pyplot.gca().get_title(),
                         Ntest_result.name)

    def test_MultiNTestPlot(self, show=False):

        n_plots = numpy.random.randint(1, 20)
        Ntests = []
        for n in range(n_plots):
            Ntest_result = mock.Mock()
            Ntest_result.name = 'Mock NTest'
            Ntest_result.sim_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(8))
            Ntest_result.test_distribution = ['poisson',
                                              numpy.random.randint(0, 20)]
            Ntest_result.observed_statistic = numpy.random.randint(0, 20)
            Ntests.append(Ntest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntests)
        Ntests.reverse()

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Ntests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntests[0].name)
        if show:
            matplotlib.pyplot.show()

    def test_MultiSTestPlot(self, show=False):

        s_plots = numpy.random.randint(1, 20)
        Stests = []
        for n in range(s_plots):
            Stest_result = mock.Mock()  # Mock class with random attributes
            Stest_result.name = 'Mock STest'
            Stest_result.sim_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(8))
            Stest_result.test_distribution = numpy.random.uniform(-1000, 0,
                                                                  numpy.random.randint(
                                                                      3,
                                                                      500)).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(-1000,
                                                                   0)  # random observed statistic
            if numpy.random.random() < 0.02:  # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        plots.plot_poisson_consistency_test(Stests)
        Stests.reverse()
        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Stests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Stests[0].name)

    def test_MultiTTestPlot(self, show=False):

        for i in range(10):
            t_plots = numpy.random.randint(2, 20)
            t_tests = []

            def rand(limit=10, offset=0):
                return limit * (numpy.random.random() - offset)

            for n in range(t_plots):
                t_result = mock.Mock()  # Mock class with random attributes
                t_result.name = 'CSEP1 Comparison Test'
                t_result.sim_name = (
                    ''.join(random.choice(string.ascii_letters)
                            for _ in range(8)), 'ref')
                t_result.observed_statistic = rand(offset=0.5)
                t_result.test_distribution = [
                    t_result.observed_statistic - rand(5),
                    t_result.observed_statistic + rand(5)]

                if numpy.random.random() < 0.05:  # sim possible infinite values
                    t_result.observed_statistic = -numpy.inf
                t_tests.append(t_result)
            matplotlib.pyplot.close()
            plots.plot_comparison_test(t_tests)
            t_tests.reverse()
            self.assertEqual(
                [i.get_text() for i in
                 matplotlib.pyplot.gca().get_xticklabels()],
                [i.sim_name[0] for i in t_tests[::-1]])
            self.assertEqual(matplotlib.pyplot.gca().get_title(),
                             t_tests[0].name)


if __name__ == '__main__':
    unittest.main()
