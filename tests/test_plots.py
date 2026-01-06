import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot
from unittest import mock
import unittest
import random
import string
from csep.utils import plots
import csep
from csep.core.catalogs import CSEPCatalog


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


class TestAlarmBasedPlots(unittest.TestCase):
    """Tests for alarm-based evaluation plots (ROC and Molchan diagrams)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        # Load forecast
        forecast_path = 'tests/artifacts/example_csep1_forecasts/Forecast/EEPAS-0F_12_1_2007.dat'
        cls.forecast = csep.load_gridded_forecast(forecast_path, name='EEPAS-0F')

        # Create synthetic catalog with events
        n_events = 15
        bbox = cls.forecast.region.get_bbox()
        min_lon, max_lon = bbox[0], bbox[1]
        min_lat, max_lat = bbox[2], bbox[3]

        # Generate random events within region bounds
        import time
        current_time = int(time.time() * 1000)
        lons = numpy.random.uniform(min_lon, max_lon, n_events)
        lats = numpy.random.uniform(min_lat, max_lat, n_events)
        magnitudes = numpy.random.uniform(4.0, 6.0, n_events)
        depths = numpy.random.uniform(0, 30, n_events)

        # Create catalog data
        catalog_data = [(i, current_time + i*1000, float(lats[i]), float(lons[i]),
                         float(depths[i]), float(magnitudes[i]))
                        for i in range(n_events)]

        cls.catalog = CSEPCatalog(data=catalog_data, region=cls.forecast.region)

    def test_plot_ROC_diagram_linear(self):
        """Test plot_ROC_diagram with linear x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_ROC_diagram_log(self):
        """Test plot_ROC_diagram with logarithmic x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_Molchan_diagram_linear(self):
        """Test plot_Molchan_diagram with linear x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_Molchan_diagram_log(self):
        """Test plot_Molchan_diagram with logarithmic x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_ROC_diagram_with_custom_axes(self):
        """Test ROC diagram can use custom axes."""
        matplotlib.pyplot.close()
        fig, ax = matplotlib.pyplot.subplots()
        result_ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            axes=ax, linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertEqual(ax, result_ax)
        matplotlib.pyplot.close()

    def test_Molchan_diagram_with_custom_axes(self):
        """Test Molchan diagram can use custom axes."""
        matplotlib.pyplot.close()
        fig, ax = matplotlib.pyplot.subplots()
        result_ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            axes=ax, linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertEqual(ax, result_ax)
        matplotlib.pyplot.close()

    def test_ROC_diagram_region_mismatch(self):
        """Test that ROC diagram raises error when regions don't match."""
        # Create catalog with different region
        from csep.core.regions import california_relm_region
        catalog_data = [(0, 1000, 35.0, -120.0, 10.0, 5.0)]
        bad_catalog = CSEPCatalog(data=catalog_data, region=california_relm_region())

        matplotlib.pyplot.close()
        with self.assertRaises(RuntimeError):
            plots.plot_ROC_diagram(
                self.forecast, bad_catalog,
                linear=True, show=False, savepdf=False, savepng=False
            )
        matplotlib.pyplot.close()

    def test_Molchan_diagram_region_mismatch(self):
        """Test that Molchan diagram raises error when regions don't match."""
        # Create catalog with different region
        from csep.core.regions import california_relm_region
        catalog_data = [(0, 1000, 35.0, -120.0, 10.0, 5.0)]
        bad_catalog = CSEPCatalog(data=catalog_data, region=california_relm_region())

        matplotlib.pyplot.close()
        with self.assertRaises(RuntimeError):
            plots.plot_Molchan_diagram(
                self.forecast, bad_catalog,
                linear=True, show=False, savepdf=False, savepng=False
            )
        matplotlib.pyplot.close()


class TestAlarmForecastLoading(unittest.TestCase):
    """Tests for alarm forecast loading and evaluation."""

    def test_load_alarm_forecast_basic(self):
        """Test loading a basic alarm forecast CSV."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        forecast = csep.load_alarm_forecast(forecast_path, name='TestAlarm')

        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.name, 'TestAlarm')
        self.assertEqual(forecast.data.shape[0], 10)  # 10 cells
        self.assertEqual(forecast.data.shape[1], 1)   # 1 magnitude bin

    def test_load_alarm_forecast_with_score_field(self):
        """Test loading alarm forecast with different score fields."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'

        # Test with alarm_score (default)
        forecast1 = csep.load_alarm_forecast(forecast_path, score_field='alarm_score')
        self.assertIsNotNone(forecast1)

        # Test with probability
        forecast2 = csep.load_alarm_forecast(forecast_path, score_field='probability')
        self.assertIsNotNone(forecast2)

        # Test with rate_per_day
        forecast3 = csep.load_alarm_forecast(forecast_path, score_field='rate_per_day')
        self.assertIsNotNone(forecast3)

        # Scores should be different
        self.assertFalse(numpy.array_equal(forecast1.data, forecast2.data))
        self.assertFalse(numpy.array_equal(forecast2.data, forecast3.data))

    def test_load_alarm_forecast_region(self):
        """Test that alarm forecast creates correct spatial region."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        forecast = csep.load_alarm_forecast(forecast_path)

        # Check region is created
        self.assertIsNotNone(forecast.region)

        # Check number of spatial bins matches data
        self.assertEqual(forecast.region.num_nodes, 10)

    def test_load_alarm_forecast_magnitudes(self):
        """Test that alarm forecast extracts correct magnitude bins."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        forecast = csep.load_alarm_forecast(forecast_path)

        # Check magnitudes are read from CSV
        self.assertEqual(forecast.magnitudes[0], 4.0)
        self.assertEqual(forecast.magnitudes[1], 6.0)

    def test_load_alarm_forecast_with_catalog_evaluation(self):
        """Test evaluating alarm forecast with synthetic catalog."""
        matplotlib.pyplot.close()

        # Load alarm forecast
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        forecast = csep.load_alarm_forecast(forecast_path, name='AlarmTest')

        # Create synthetic catalog in same region
        n_events = 5
        bbox = forecast.region.get_bbox()
        min_lon, max_lon = bbox[0], bbox[1]
        min_lat, max_lat = bbox[2], bbox[3]

        import time
        current_time = int(time.time() * 1000)
        lons = numpy.random.uniform(min_lon, max_lon, n_events)
        lats = numpy.random.uniform(min_lat, max_lat, n_events)
        magnitudes = numpy.random.uniform(4.0, 6.0, n_events)
        depths = numpy.random.uniform(0, 30, n_events)

        catalog_data = [(i, current_time + i*1000, float(lats[i]), float(lons[i]),
                         float(depths[i]), float(magnitudes[i]))
                        for i in range(n_events)]

        catalog = CSEPCatalog(data=catalog_data, region=forecast.region)

        # Test ROC diagram
        ax = plots.plot_ROC_diagram(
            forecast, catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

        # Test Molchan diagram
        ax = plots.plot_Molchan_diagram(
            forecast, catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_load_alarm_forecast_missing_file(self):
        """Test that loading non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            csep.load_alarm_forecast('nonexistent_file.csv')

    def test_load_alarm_forecast_invalid_score_field(self):
        """Test that invalid score field raises error."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        with self.assertRaises(ValueError):
            csep.load_alarm_forecast(forecast_path, score_field='invalid_column')

    def test_load_alarm_forecast_tab_delimiter(self):
        """Test loading alarm forecast with tab delimiter."""
        forecast_path = 'tests/artifacts/alarm_forecast_tabs.tsv'
        forecast = csep.load_alarm_forecast(forecast_path, delimiter='\t')
        
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.data.shape[0], 10)  # 10 cells
        # Check that scores were loaded correctly
        self.assertAlmostEqual(forecast.data[0, 0], 0.85, places=2)

    def test_load_alarm_forecast_semicolon_delimiter(self):
        """Test loading alarm forecast with semicolon delimiter."""
        forecast_path = 'tests/artifacts/alarm_forecast_semicolon.csv'
        forecast = csep.load_alarm_forecast(forecast_path, delimiter=';')
        
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.data.shape[0], 10)  # 10 cells

    def test_load_alarm_forecast_multiple_score_fields(self):
        """Test loading alarm forecast with multiple score fields."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        forecast = csep.load_alarm_forecast(
            forecast_path,
            score_fields=['alarm_score', 'probability', 'rate_per_day']
        )
        
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.data.shape[0], 10)  # 10 cells
        self.assertEqual(forecast.data.shape[1], 3)   # 3 score fields
        
        # Check first row values match expected
        self.assertAlmostEqual(forecast.data[0, 0], 0.85, places=2)  # alarm_score
        self.assertAlmostEqual(forecast.data[0, 1], 0.75, places=2)  # probability
        self.assertAlmostEqual(forecast.data[0, 2], 0.002, places=4)  # rate_per_day

    def test_load_alarm_forecast_custom_magnitude_bins(self):
        """Test loading alarm forecast with custom magnitude bins."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        custom_bins = [4.0, 5.0, 6.0, 7.0]
        forecast = csep.load_alarm_forecast(
            forecast_path,
            magnitude_bins=custom_bins
        )
        
        self.assertIsNotNone(forecast)
        numpy.testing.assert_array_equal(forecast.magnitudes, custom_bins)

    def test_load_alarm_forecast_invalid_magnitude_bins(self):
        """Test that invalid magnitude bins raises error."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        with self.assertRaises(ValueError):
            csep.load_alarm_forecast(forecast_path, magnitude_bins=[4.0])  # Need at least 2 edges

    def test_load_alarm_forecast_invalid_score_fields(self):
        """Test that invalid score_fields type raises error."""
        forecast_path = 'tests/artifacts/alarm_forecast_example.csv'
        with self.assertRaises(ValueError):
            csep.load_alarm_forecast(forecast_path, score_fields='not_a_list')


class TestTemporalEvaluation(unittest.TestCase):
    """Tests for temporal probability forecast evaluation (ROC and Molchan)."""

    def test_load_temporal_forecast_basic(self):
        """Test loading a basic temporal forecast CSV."""
        forecast_path = 'tests/artifacts/temporal_forecast_example.csv'
        data = csep.load_temporal_forecast(forecast_path)

        self.assertIn('times', data)
        self.assertIn('probabilities', data)
        self.assertIn('metadata', data)
        self.assertNotIn('observations', data)  # Observations not in CSV anymore

        self.assertEqual(len(data['probabilities']), 19)  # 19 time windows

    def test_load_temporal_forecast_missing_file(self):
        """Test that loading non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            csep.load_temporal_forecast('nonexistent_file.csv')

    def test_temporal_ROC_diagram_basic(self):
        """Test basic temporal ROC diagram."""
        matplotlib.pyplot.close()

        # Simple test data
        probs = numpy.array([0.9, 0.7, 0.5, 0.3, 0.1])
        obs = numpy.array([1, 1, 0, 0, 0])

        ax, auc = plots.plot_temporal_ROC_diagram(
            probs, obs, name='TestForecast', show=False,
            savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        self.assertGreater(auc, 0.5)  # Should be better than random
        matplotlib.pyplot.close()

    def test_temporal_ROC_diagram_perfect_forecast(self):
        """Test ROC diagram with perfect forecast."""
        matplotlib.pyplot.close()

        # Perfect forecast: all events have highest probabilities
        probs = numpy.array([1.0, 0.9, 0.0, 0.0])
        obs = numpy.array([1, 1, 0, 0])

        ax, auc = plots.plot_temporal_ROC_diagram(
            probs, obs, show=False, savepdf=False, savepng=False
        )
        self.assertAlmostEqual(auc, 1.0, places=1)  # AUC should be ~1
        matplotlib.pyplot.close()

    def test_temporal_Molchan_diagram_basic(self):
        """Test basic temporal Molchan diagram."""
        matplotlib.pyplot.close()

        # Simple test data
        probs = numpy.array([0.9, 0.7, 0.5, 0.3, 0.1])
        obs = numpy.array([1, 1, 0, 0, 0])

        ax, ass, ass_std = plots.plot_temporal_Molchan_diagram(
            probs, obs, name='TestForecast', show=False,
            savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        self.assertGreater(ass, 0.5)  # Should be better than random
        self.assertGreater(ass_std, 0)  # Should have uncertainty estimate
        matplotlib.pyplot.close()

    def test_temporal_Molchan_diagram_perfect_forecast(self):
        """Test Molchan diagram with a good forecast (events have highest probabilities)."""
        matplotlib.pyplot.close()

        # Good forecast: events have highest probs, many non-event days
        # More non-event days = higher ASS if events still at top
        probs = numpy.array([0.95, 0.90, 0.10, 0.08, 0.06, 0.04, 0.02])
        obs = numpy.array([1, 1, 0, 0, 0, 0, 0])

        ax, ass, ass_std = plots.plot_temporal_Molchan_diagram(
            probs, obs, show=False, savepdf=False, savepng=False
        )
        # With 2 events in 7 days, if both events are at highest probs:
        # At threshold 0.10: tau=2/7≈0.29, nu=0 -> trajectory goes quickly to nu=0
        # This should give ASS > 0.5
        self.assertGreater(ass, 0.5)  # Should be better than random
        matplotlib.pyplot.close()

    def test_temporal_diagrams_with_loaded_forecast(self):
        """Test temporal diagrams with loaded CSV data and synthetic observations."""
        matplotlib.pyplot.close()

        # Load actual forecast data
        forecast_path = 'tests/artifacts/temporal_forecast_example.csv'
        data = csep.load_temporal_forecast(forecast_path)

        # Create synthetic observations (events on days 3, 15, 17 - matching original)
        observations = numpy.zeros(19, dtype=int)
        observations[2] = 1   # day 3
        observations[14] = 1  # day 15
        observations[16] = 1  # day 17

        # Test ROC
        ax1, auc = plots.plot_temporal_ROC_diagram(
            data['probabilities'], observations,
            name='DailyM4Forecast', show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax1)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)
        matplotlib.pyplot.close()

        # Test Molchan
        ax2, ass, sigma = plots.plot_temporal_Molchan_diagram(
            data['probabilities'], observations,
            name='DailyM4Forecast', show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax2)
        self.assertGreaterEqual(ass, 0.0)
        self.assertLessEqual(ass, 1.0)
        matplotlib.pyplot.close()

    def test_temporal_diagrams_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        probs = numpy.array([0.5, 0.3])
        obs = numpy.array([1, 0, 0])  # Different length

        with self.assertRaises(ValueError):
            plots.plot_temporal_ROC_diagram(probs, obs, show=False)

        with self.assertRaises(ValueError):
            plots.plot_temporal_Molchan_diagram(probs, obs, show=False)

    def test_temporal_diagrams_log_scale(self):
        """Test temporal diagrams with logarithmic x-axis."""
        matplotlib.pyplot.close()

        probs = numpy.array([0.9, 0.7, 0.5, 0.3, 0.1])
        obs = numpy.array([1, 1, 0, 0, 0])

        ax, auc = plots.plot_temporal_ROC_diagram(
            probs, obs, linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

        ax, ass, sigma = plots.plot_temporal_Molchan_diagram(
            probs, obs, linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()


if __name__ == '__main__':
    unittest.main()
