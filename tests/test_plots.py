import numpy
import matplotlib.pyplot
from unittest import mock
import unittest
import random
import string
from csep.utils import plots
import pytest

def random_region(n):

    n_iter = 0
    while n_iter < 100:
        x_1 = numpy.random.uniform(-180, 179)
        y_1 = numpy.random.uniform(-90, 89)
        x_2 = numpy.random.uniform(x_1 + 0.001, 180)
        y_2 = numpy.random.uniform(y_1 + 0.001, 90)
        xs = numpy.linspace(x_1, x_2, n)
        dh = xs[1] - xs[0]
        ys = numpy.arange(y_1, y_2, dh)

        bbox = [min(xs), max(xs), min(ys), max(ys)]
        if   10 < ys.shape[0] < 1000 and max(ys) + dh < 90 \
                and max(xs) + dh < 180:
            break
        n_iter += 1
    return (xs, ys, dh, bbox)

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
            Stest_result.test_distribution = numpy.random.uniform(-1000, 0, numpy.random.randint(500)).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(-1000, 0) #random observed statistic
            if numpy.random.random() <0.02: # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Stests)

        self.assertEqual(matplotlib.pyplot.gca().collections, ax.collections)
        self.assertEqual([i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                        [i.sim_name for i in Stests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Stests[0].name)

class TestSpatialPlot(unittest.TestCase):

    def test_SpatialDataset(self, show=False):
        # Get random region between x in [-180,180] and y in [-90, 90]
        xs, ys, dh, bbox = random_region(numpy.random.randint(10,50))
        # Create mock class and methods
        region = mock.Mock()
        region.get_bbox = mock.Mock()
        region.get_bbox.return_value = bbox
        region.xs = xs
        region.ys = ys
        region.dh = dh
        show_flag = bool(random.getrandbits(1))
        # Create random dataset
        gridded = numpy.random.random((ys.shape[0], xs.shape[0]))
        # Call func
        ax = plots.plot_spatial_dataset(gridded, region, show=show_flag,
                                        plot_args={'grid': True, 'coastline':True})
        # Assert extent of the plot
        for i,j in zip(ax.get_xbound(), bbox[:2]):
            self.assertAlmostEqual(i, j)
        for i,j in zip(ax.get_ybound(), bbox[2:]):
            self.assertAlmostEqual(i, j + dh)
        # assert ax elements. 3 elements if plot showed, 1 if not, because cartopy does not draw unless show is called.
        self.assertEqual(len(ax.collections), 1 + 2*show_flag)
        self.assertEqual(ax.collections[0].get_array().shape[0], gridded.size)


if __name__ == '__main__':
    unittest.main()
    # pytest.main()
