import copy
import gc
import json
import os.path
import random
import socket
import string
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, Mock

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy
from cartopy import crs as ccrs
from matplotlib import colors

import csep
from csep.core import catalogs
from csep.core.catalog_evaluations import (
    CatalogNumberTestResult,
    CatalogSpatialTestResult,
    CatalogMagnitudeTestResult,
    CatalogPseudolikelihoodTestResult,
    CalibrationTestResult,
)
from csep.utils.plots import (
    plot_cumulative_events_versus_time,
    plot_magnitude_versus_time,
    plot_distribution_test,
    plot_magnitude_histogram,
    plot_calibration_test,
    plot_comparison_test,
    plot_consistency_test,
    plot_basemap,
    plot_catalog,
    plot_gridded_dataset,
    plot_concentration_ROC_diagram,
    plot_Molchan_diagram,
    plot_ROC_diagram,
    _get_basemap,  # noqa
    _calculate_spatial_extent,  # noqa
    _create_geo_axes,  # noqa
    _add_gridlines,  # noqa
    _get_marker_style,  # noqa
    _get_marker_t_color,  # noqa
    _get_marker_w_color,  # noqa
    _get_axis_limits,  # noqa
    _autosize_scatter,  # noqa
    _autoscale_histogram,  # noqa
    _annotate_distribution_plot,  # noqa
    _get_colormap,  # noqa
    _process_stat_distribution,  # noqa
)


def is_internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def is_github_ci():
    if os.getenv("GITHUB_ACTIONS") or os.getenv("CI") or os.getenv("GITHUB_ACTION"):
        return True
    else:
        return False


show_plots = False


class TestPlots(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent constructor
        # Define the save directory
        self.artifacts = os.path.join(os.path.dirname(__file__), "artifacts")
        self.save_dir = os.path.join(self.artifacts, "plots")
        os.makedirs(self.save_dir, exist_ok=True)

    def savefig(self, ax, name):
        ax.figure.savefig(os.path.join(self.save_dir, name))


#
class TestTimeSeriesPlots(TestPlots):

    def setUp(self):
        # This method is called before each test.
        # Load the stochastic event sets and observation here.

        cat_file_m2 = csep.datasets.comcat_example_catalog_fname
        cat_file_m5 = os.path.join(
            self.artifacts,
            "example_csep2_forecasts",
            "Catalog",
            "catalog.json",
        )

        forecast_file = os.path.join(
            self.artifacts,
            "example_csep2_forecasts",
            "Forecasts",
            "ucerf3-landers_short.csv",
        )

        self.stochastic_event_sets = csep.load_catalog_forecast(forecast_file)
        self.observation_m5 = catalogs.CSEPCatalog.load_json(cat_file_m5)
        self.observation_m2 = csep.load_catalog(cat_file_m2)

    def test_plot_magnitude_vs_time(self):
        # Basic test
        ax = plot_magnitude_versus_time(catalog=self.observation_m2, show=show_plots)
        self.assertEqual(ax.get_title(), "")
        self.assertEqual(ax.get_xlabel(), "Datetime")
        self.assertEqual(ax.get_ylabel(), "Magnitude")

        # Test with custom color
        ax = plot_magnitude_versus_time(catalog=self.observation_m2, color="red", show=show_plots)
        scatter_color = ax.collections[0].get_facecolor()[0]
        self.assertTrue(all(scatter_color[:3] == (1.0, 0.0, 0.0)))  # Check if color is red

        # Test with custom marker size
        ax = plot_magnitude_versus_time(
            catalog=self.observation_m2, size=25, max_size=600, show=show_plots
        )
        scatter_sizes = ax.collections[0].get_sizes()
        func_sizes = _autosize_scatter(self.observation_m2.data["magnitude"], 25, 600, 4)
        numpy.testing.assert_array_almost_equal(scatter_sizes, func_sizes)

        # Test with custom alpha
        ax = plot_magnitude_versus_time(catalog=self.observation_m2, alpha=0.5, show=show_plots)
        scatter_alpha = ax.collections[0].get_alpha()
        self.assertEqual(scatter_alpha, 0.5)

        # Test with custom marker size power
        ax = plot_magnitude_versus_time(catalog=self.observation_m2, power=6, show=show_plots)
        scatter_sizes = ax.collections[0].get_sizes()
        func_sizes = _autosize_scatter(self.observation_m2.data["magnitude"], 4, 300, 6)
        numpy.testing.assert_array_almost_equal(scatter_sizes, func_sizes)
        #
        # # Test with show=True (just to ensure no errors occur)
        plot_magnitude_versus_time(catalog=self.observation_m2, show=False)
        plt.close("all")

    def test_plot_cumulative_events_default(self):
        # Test with default arguments to ensure basic functionality
        ax = plot_cumulative_events_versus_time(
            catalog_forecast=self.stochastic_event_sets,
            observation=self.observation_m5,
            show=show_plots,
        )

        self.assertIsNotNone(ax.get_title())
        self.assertIsNotNone(ax.get_xlabel())
        self.assertIsNotNone(ax.get_ylabel())

    def test_plot_cumulative_events_hours(self):
        # Test with time_axis set to 'hours'
        ax = plot_cumulative_events_versus_time(
            catalog_forecast=self.stochastic_event_sets,
            observation=self.observation_m5,
            bins=50,
            time_axis="hours",
            xlabel="Hours since Mainshock",
            ylabel="Cumulative Event Count",
            title="Cumulative Event Counts by Hour",
            legend_loc="upper left",
            show=show_plots,
        )

        self.assertEqual(ax.get_xlabel(), "Hours since Mainshock")
        self.assertEqual(ax.get_ylabel(), "Cumulative Event Count")
        self.assertEqual(ax.get_title(), "Cumulative Event Counts by Hour")

    def test_plot_cumulative_events_different_bins(self):
        # Test with different number of bins
        ax = plot_cumulative_events_versus_time(
            catalog_forecast=self.stochastic_event_sets,
            observation=self.observation_m5,
            bins=200,
            show=show_plots,
            figsize=(12, 8),
            time_axis="days",
            xlabel="Days since Mainshock",
            ylabel="Cumulative Event Count",
            title="Cumulative Event Counts with More Bins",
            legend_loc="best",
        )

        self.assertEqual(ax.get_title(), "Cumulative Event Counts with More Bins")
        self.assertEqual(ax.get_xlabel(), "Days since Mainshock")
        self.assertEqual(ax.get_ylabel(), "Cumulative Event Count")

    def test_plot_cumulative_events_custom_legend(self):
        # Test with a custom legend location and size
        ax = plot_cumulative_events_versus_time(
            catalog_forecast=self.stochastic_event_sets,
            observation=self.observation_m5,
            bins=75,
            show=show_plots,
            figsize=(8, 5),
            time_axis="days",
            xlabel="Days since Mainshock",
            ylabel="Cumulative Event Count",
            title="Cumulative Event Counts with Custom Legend",
            legend_loc="lower right",
            legend_fontsize=14,
        )

        self.assertEqual(ax.get_legend()._get_loc(), 4)
        self.assertEqual(ax.get_legend().get_texts()[0].get_fontsize(), 14)

    def tearDown(self):
        plt.close("all")
        del self.stochastic_event_sets
        del self.observation_m2
        del self.observation_m5
        gc.collect()


class TestPlotMagnitudeHistogram(TestPlots):

    def setUp(self):

        def gr_dist(num_events, mag_min=3.0, mag_max=8.0, b_val=1.0):
            U = numpy.random.uniform(0, 1, num_events)
            magnitudes = mag_min - (1.0 / b_val) * numpy.log10(1 - U)
            magnitudes = magnitudes[magnitudes <= mag_max]
            return magnitudes

        self.mock_forecast = [MagicMock(), MagicMock(), MagicMock()]
        for i in self.mock_forecast:
            i.get_magnitudes.return_value = gr_dist(5000)

        self.mock_cat = MagicMock()
        self.mock_cat.get_magnitudes.return_value = gr_dist(500, b_val=1.2)
        self.mock_cat.get_number_of_events.return_value = 500
        self.mock_cat.region.magnitudes = numpy.arange(3.0, 8.0, 0.1)

        cat_file_m5 = os.path.join(
            self.artifacts,
            "example_csep2_forecasts",
            "Catalog",
            "catalog.json",
        )
        self.comcat = catalogs.CSEPCatalog.load_json(cat_file_m5)
        forecast_file = os.path.join(
            self.artifacts,
            "example_csep2_forecasts",
            "Forecasts",
            "ucerf3-landers_short.csv",
        )

        self.stochastic_event_sets = csep.load_catalog_forecast(forecast_file)

        os.makedirs(self.save_dir, exist_ok=True)

    def test_plot_magnitude_histogram_basic(self):
        # Test with basic arguments
        plot_magnitude_histogram(
            self.mock_forecast, self.mock_cat, show=show_plots, density=True
        )

        # Verify that magnitudes were retrieved
        for catalog in self.mock_forecast:
            catalog.get_magnitudes.assert_called_once()
        self.mock_cat.get_magnitudes.assert_called_once()
        self.mock_cat.get_number_of_events.assert_called_once()

    def test_plot_magnitude_histogram_ucerf(self):
        # Test with basic arguments
        plot_magnitude_histogram(self.stochastic_event_sets, self.comcat, show=show_plots)

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestPlotDistributionTests(TestPlots):

    def setUp(self):
        self.result_obs_scalar = MagicMock()
        self.result_obs_scalar.test_distribution = numpy.random.normal(0, 1, 1000)
        self.result_obs_scalar.observed_statistic = numpy.random.rand(1)[0]

        self.result_obs_array = MagicMock()
        self.result_obs_array.test_distribution = numpy.random.normal(0, 1, 1000)
        self.result_obs_array.observed_statistic = numpy.random.normal(0, 1, 100)

        self.result_nan = MagicMock()
        self.result_nan.test_distribution = numpy.random.normal(0, 1, 1000)
        self.result_nan.observed_statistic = -numpy.inf

        # Example data for testing
        n_test = os.path.join(
            self.artifacts, "example_csep2_forecasts", "Results", "catalog_n_test.json"
        )
        s_test = os.path.join(
            self.artifacts, "example_csep2_forecasts", "Results", "catalog_s_test.json"
        )
        m_test = os.path.join(
            self.artifacts, "example_csep2_forecasts", "Results", "catalog_m_test.json"
        )
        l_test = os.path.join(
            self.artifacts, "example_csep2_forecasts", "Results", "catalog_l_test.json"
        )

        with open(n_test, "r") as fp:
            self.n_test = CatalogNumberTestResult.from_dict(json.load(fp))
        with open(s_test, "r") as fp:
            self.s_test = CatalogSpatialTestResult.from_dict(json.load(fp))
        with open(m_test, "r") as fp:
            self.m_test = CatalogMagnitudeTestResult.from_dict(json.load(fp))
        with open(l_test, "r") as fp:
            self.l_test = CatalogPseudolikelihoodTestResult.from_dict(json.load(fp))

    def test_plot_dist_test_with_scalar_observation_default(self):
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_scalar,
            show=show_plots,
        )

        # Check if a vertical line was drawn for the scalar observation
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        self.assertEqual(len(lines), 1)  # Expect one vertical line
        self.assertEqual(lines[0].get_xdata()[0], self.result_obs_scalar.observed_statistic)

    def test_plot_dist_test_with_scalar_observation_w_labels(self):
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_scalar,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            show=show_plots,
        )

        # Check if a vertical line was drawn for the scalar observation
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        self.assertEqual(len(lines), 1)  # Expect one vertical line
        self.assertEqual(lines[0].get_xdata()[0], self.result_obs_scalar.observed_statistic)

    def test_plot_dist_test_with_array_observation(self):
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_array,
            alpha=0.5,
            show=show_plots,
        )
        bars = ax.patches
        self.assertTrue(
            all(bar.get_alpha() == 0.5 for bar in bars),
            "Alpha transparency not set correctly for bars",
        )

    def test_plot_dist_test_with_percentile_shading(self):
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_scalar,
            percentile=60,
            show=show_plots,
        )
        expected_red = (1.0, 0.0, 0.0)
        red_patches = []
        for patch_ in ax.patches:
            facecolor = patch_.get_facecolor()[:3]  # Get RGB, ignore alpha
            if all(abs(facecolor[i] - expected_red[i]) < 0.01 for i in range(3)):
                red_patches.append(patch_)
        self.assertGreater(
            len(red_patches),
            0,
            "Expected some patches to be colored red for percentile shading",
        )

    def test_plot_dist_test_with_annotation(self):
        annotation_text = "Test Annotation"
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_scalar,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            annotation_text=annotation_text,
            annotation_xy=(0.5, 0.5),
            annotation_fontsize=12,
            show=show_plots,
        )
        annotations = ax.texts
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].get_text(), annotation_text)

    def test_plot_dist_test_xlim(self):
        xlim = (-5, 5)
        ax = plot_distribution_test(
            evaluation_result=self.result_obs_scalar,
            percentile=95,
            xlim=xlim,
            show=show_plots,
        )
        self.assertEqual(ax.get_xlim(), xlim)

    def test_plot_dist_test_autoxlim_nan(self):

        plot_distribution_test(
            evaluation_result=self.result_nan,
            percentile=95,
            show=show_plots,
        )

    def test_plot_n_test(self):
        plot_distribution_test(
            self.n_test,
            show=show_plots,
        )

    def test_plot_m_test(self):
        plot_distribution_test(
            self.m_test,
            show=show_plots,
        )

    def test_plot_s_test(self):
        plot_distribution_test(
            self.s_test,
            show=show_plots,
        )

    def test_plot_l_test(self):
        plot_distribution_test(
            self.l_test,
            show=show_plots,
        )

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestPlotCalibrationTest(TestPlots):

    def setUp(self):
        # Create a mock evaluation result with a uniform distribution
        self.evaluation_result = MagicMock()
        self.evaluation_result.test_distribution = numpy.random.uniform(0, 1, 1000) ** 1.3
        self.evaluation_result.sim_name = "Simulated Data"

        # Example data for testing
        cal_n_test = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            "example_csep2_forecasts",
            "Results",
            "calibration_n.json",
        )
        cal_m_test = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            "example_csep2_forecasts",
            "Results",
            "calibration_m.json",
        )

        with open(cal_n_test, "r") as fp:
            self.cal_n_test = CalibrationTestResult.from_dict(json.load(fp))
        with open(cal_m_test, "r") as fp:
            self.cal_m_test = CalibrationTestResult.from_dict(json.load(fp))

    def test_plot_calibration_basic(self):
        # Test with basic arguments
        ax = plot_calibration_test(self.evaluation_result, show=show_plots)
        # Check if the plot was created
        self.assertIsInstance(ax, plt.Axes)
        # Check if the confidence intervals were plotted (3 lines: pp, ulow, uhigh)
        self.assertEqual(len(ax.lines), 4)
        # Check if the legend was created with the correct label
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        legend_labels = [text.get_text() for text in legend.get_texts()]
        self.assertIn(self.evaluation_result.sim_name, legend_labels)

    def test_plot_calibration_test_n_test(self):

        ax = plot_calibration_test(self.cal_n_test, show=show_plots)
        self.savefig(ax, "calibration_n_test.png")
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        legend_labels = [text.get_text() for text in legend.get_texts()]
        self.assertIn(self.cal_n_test.sim_name, legend_labels)

    def test_plot_calibration_test_m_test(self):
        ax = plot_calibration_test(self.cal_m_test, show=show_plots)
        self.savefig(ax, "calibration_m_test.png")
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        legend_labels = [text.get_text() for text in legend.get_texts()]
        self.assertIn(self.cal_m_test.sim_name, legend_labels)

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestBatchPlots(TestPlots):
    def setUp(self):
        # Mocking EvaluationResult for testing
        self.mock_result = Mock()
        self.mock_result.sim_name = "Mock Forecast"
        self.mock_result.test_distribution = numpy.random.normal(loc=10, scale=2, size=100)
        self.mock_result.observed_statistic = 8

    def test_plot_consistency_basic(self):
        ax = plot_consistency_test(eval_results=self.mock_result, show=show_plots)
        self.assertEqual(ax.get_title(), "")
        self.assertEqual(ax.get_xlabel(), "Statistic distribution")

    def test_plot_consistency_with_multiple_results(self):
        mock_results = [self.mock_result for _ in range(5)]
        ax = plot_consistency_test(eval_results=mock_results, show=show_plots)
        self.assertEqual(len(ax.get_yticklabels()), 5)

    def test_plot_consistency_with_normalization(self):
        ax = plot_consistency_test(
            eval_results=self.mock_result, normalize=True, show=show_plots
        )
        # Assert that the observed statistic is plotted at 0
        self.assertEqual(ax.lines[0].get_xdata(), 0)

    def test_plot_consistency_with_one_sided_lower(self):
        mock_result = copy.deepcopy(self.mock_result)
        # THe observed statistic is placed to the right of the model test distribution.
        mock_result.observed_statistic = max(self.mock_result.test_distribution) + 1
        ax = plot_consistency_test(
            eval_results=mock_result, one_sided_lower=True, show=show_plots
        )
        # The end of the infinite dashed line should extend way away from the plot limit
        self.assertGreater(ax.lines[-1].get_xdata()[-1], ax.get_xlim()[1])

    def test_plot_consistency_with_custom_percentile(self):
        ax = plot_consistency_test(
            eval_results=self.mock_result, percentile=99, show=show_plots
        )

        # Check that the line extent equals the lower 0.5 % percentile
        self.assertAlmostEqual(
            ax.lines[2].get_xdata(), numpy.percentile(self.mock_result.test_distribution, 0.5)
        )

    def test_plot_consistency_with_variance(self):
        mock_nb = copy.deepcopy(self.mock_result)
        mock_poisson = copy.deepcopy(self.mock_result)
        mock_nb.test_distribution = ("negative_binomial", 8, 16)
        mock_poisson.test_distribution = ("poisson", 8)
        ax_nb = plot_consistency_test(eval_results=mock_nb, variance=16, show=show_plots)
        ax_p = plot_consistency_test(eval_results=mock_poisson, variance=None, show=show_plots)
        # Ensure the negative binomial has a larger x-axis extent than poisson
        self.assertTrue(ax_p.get_xlim()[1] < ax_nb.get_xlim()[1])

    def test_plot_consistency_with_custom_plot_args(self):
        ax = plot_consistency_test(
            eval_results=self.mock_result,
            show=show_plots,
            xlabel="Custom X",
            ylabel="Custom Y",
            title="Custom Title",
        )
        self.assertEqual(ax.get_xlabel(), "Custom X")
        self.assertEqual(ax.get_title(), "Custom Title")

    def test_plot_consistency_with_mean(self):
        ax = plot_consistency_test(
            eval_results=self.mock_result, plot_mean=True, show=show_plots
        )
        # Check for the mean line plotted as a circle
        self.assertTrue(any(["o" in str(line.get_marker()) for line in ax.lines]))

    def test_SingleNTestPlot(self):

        expected_val = numpy.random.randint(0, 20)
        observed_val = numpy.random.randint(0, 20)
        Ntest_result = mock.Mock()
        Ntest_result.name = "Mock NTest"
        Ntest_result.sim_name = "Mock SimName"
        Ntest_result.test_distribution = ["poisson", expected_val]
        Ntest_result.observed_statistic = observed_val
        matplotlib.pyplot.close()
        plot_consistency_test(Ntest_result, show=show_plots)

        if not show_plots:
            self.assertEqual(
                [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                [i.sim_name for i in [Ntest_result]],
            )
            self.assertEqual(matplotlib.pyplot.gca().get_title(), "")

    def test_MultiNTestPlot(self):

        n_plots = numpy.random.randint(1, 20)
        Ntests = []
        for n in range(n_plots):
            Ntest_result = mock.Mock()
            Ntest_result.name = "Mock NTest"
            Ntest_result.sim_name = "".join(
                random.choice(string.ascii_letters) for _ in range(8)
            )
            Ntest_result.test_distribution = ["poisson", numpy.random.randint(0, 20)]
            Ntest_result.observed_statistic = numpy.random.randint(0, 20)
            Ntests.append(Ntest_result)
        matplotlib.pyplot.close()
        plot_consistency_test(Ntests, show=show_plots)
        Ntests.reverse()
        if not show_plots:
            self.assertEqual(
                [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
                [i.sim_name for i in Ntests],
            )

    def test_MultiSTestPlot(self):

        s_plots = numpy.random.randint(1, 20)
        Stests = []
        for n in range(s_plots):
            Stest_result = mock.Mock()  # Mock class with random attributes
            Stest_result.name = "Mock STest"
            Stest_result.sim_name = "".join(
                random.choice(string.ascii_letters) for _ in range(8)
            )
            Stest_result.test_distribution = numpy.random.uniform(
                -1000, 0, numpy.random.randint(3, 500)
            ).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(
                -1000, 0
            )  # random observed statistic
            if numpy.random.random() < 0.02:  # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        plot_consistency_test(Stests)
        Stests.reverse()
        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Stests],
        )

    def test_MultiTTestPlot(self):

        for i in range(1):
            t_plots = numpy.random.randint(2, 20)
            t_tests = []

            def rand(limit=10, offset=0.0):
                return limit * (numpy.random.random() - offset)

            for n in range(t_plots):
                t_result = mock.Mock()  # Mock class with random attributes
                t_result.name = "CSEP1 Comparison Test"
                t_result.sim_name = (
                    "".join(random.choice(string.ascii_letters) for _ in range(8)),
                    "ref",
                )
                t_result.observed_statistic = rand(offset=0.5)
                t_result.test_distribution = [
                    t_result.observed_statistic - rand(5),
                    t_result.observed_statistic + rand(5),
                ]
                t_result.quantile = (None, None, 0.05)
                if numpy.random.random() < 0.05:  # sim possible infinite values
                    t_result.observed_statistic = -numpy.inf
                t_tests.append(t_result)
            matplotlib.pyplot.close()
            plot_comparison_test(t_tests, show=show_plots)
            t_tests.reverse()
            if not show_plots:
                self.assertEqual(
                    [i.get_text() for i in matplotlib.pyplot.gca().get_xticklabels()],
                    [i.sim_name[0] for i in t_tests[::-1]],
                )
                self.assertEqual(matplotlib.pyplot.gca().get_title(), t_tests[0].name)

    def tearDown(self):
        plt.close("all")

        gc.collect()


class TestPlotBasemap(TestPlots):

    def setUp(self):
        self.chiloe_extent = [-75, -71, -44, -40]

    @patch("csep.utils.plots._get_basemap")
    def test_plot_basemap_default(self, mock_get_basemap):

        mock_tiles = MagicMock()
        mock_get_basemap.return_value = mock_tiles
        ax = plot_basemap(show=show_plots)
        self.assertIsInstance(ax, plt.Axes)
        mock_get_basemap.assert_not_called()

    @patch("csep.utils.plots._get_basemap")
    def test_plot_basemap_with_features(self, mock_get_basemap):
        mock_tiles = MagicMock()
        mock_get_basemap.return_value = mock_tiles

        basemap = "stock_img"
        ax = plot_basemap(
            basemap=basemap,
            extent=self.chiloe_extent,
            coastline=True,
            borders=True,
            tile_scaling=5,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
        )

        self.assertIsInstance(ax, plt.Axes)
        mock_get_basemap.assert_not_called()
        self.assertTrue(ax.get_legend() is None)

    @unittest.skipIf(is_github_ci(), "Skipping test in GitHub CI environment")
    @unittest.skipIf(not is_internet_available(), "Skipping test due to no internet connection")
    def test_plot_google_satellite(self):
        basemap = "google-satellite"
        ax = plot_basemap(
            basemap=basemap,
            extent=self.chiloe_extent,
            coastline=True,
            tile_depth=4,
            show=show_plots,
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(ax.get_legend() is None)

    @unittest.skipIf(is_github_ci(), "Skipping test in GitHub CI environment")
    @unittest.skipIf(not is_internet_available(), "Skipping test due to no internet connection")
    def test_plot_esri(self):
        basemap = "ESRI_terrain"

        ax = plot_basemap(
            basemap,
            self.chiloe_extent,
            coastline=True,
            borders=True,
            tile_depth=4,
            show=show_plots,
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(ax.get_legend() is None)

    @patch("csep.utils.plots._get_basemap")
    def test_plot_basemap_set_global(self, mock_get_basemap):
        # Mock the _get_basemap function
        mock_tiles = MagicMock()
        mock_get_basemap.return_value = mock_tiles

        # Test data for global view
        basemap = None
        ax = plot_basemap(basemap, set_global=True, show=show_plots)

        # Assertions
        self.assertIsInstance(ax, plt.Axes)
        mock_get_basemap.assert_not_called()
        self.assertTrue(ax.get_extent() == (-180, 180, -90, 90))

    @unittest.skipIf(is_github_ci(), "Skipping test in GitHub CI environment")
    def test_plot_basemap_tif_file(self):
        basemap = csep.datasets.basemap_california
        projection = ccrs.PlateCarree()
        extent = [-126, -111, 30, 42.5]
        ax = plot_basemap(basemap, extent=extent, projection=projection, show=show_plots)

        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.projection, projection)

    def test_plot_basemap_with_custom_projection(self):
        projection = ccrs.Mercator()
        basemap = None
        ax = plot_basemap(basemap, self.chiloe_extent, projection=projection, show=show_plots)

        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.projection, projection)

    def test_plot_basemap_with_custom_projection_and_features(self):
        projection = ccrs.Mercator()
        basemap = None
        ax = plot_basemap(
            basemap=basemap,
            extent=self.chiloe_extent,
            projection=projection,
            coastline=True,
            borders=True,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
        )

        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.projection, projection)

    def tearDown(self):

        plt.close("all")
        gc.collect()


class TestPlotCatalog(TestPlots):

    def setUp(self):
        # Set up a mock catalog with basic properties
        self.mock_catalog = MagicMock()

        size = numpy.random.randint(4, 20)
        self.mock_catalog.get_magnitudes.return_value = numpy.random.random(size) * 4 + 4
        self.mock_catalog.get_longitudes.return_value = numpy.random.random(size) * 30 - 120
        self.mock_catalog.get_latitudes.return_value = numpy.random.random(size) * 30 + 30
        self.mock_catalog.name = "Mock Catalog"

        self.mock_catalog.get_bbox.return_value = [
            numpy.min(self.mock_catalog.get_longitudes()),
            numpy.max(self.mock_catalog.get_longitudes()),
            numpy.min(self.mock_catalog.get_latitudes()),
            numpy.max(self.mock_catalog.get_latitudes()),
        ]

        # Mock region if needed
        self.mock_catalog.region.get_bbox.return_value = [-125, -85, 25, 65]
        self.mock_catalog.region.tight_bbox.return_value = numpy.array(
            [[-125, 25], [-85, 25], [-85, 65], [-125, 65], [-125, 25]]
        )

        self.mock_fix = MagicMock()
        self.mock_fix.get_magnitudes.return_value = numpy.array([4, 5, 6, 7, 8])
        self.mock_fix.get_latitudes.return_value = numpy.array([36, 35, 34, 33, 32])
        self.mock_fix.get_longitudes.return_value = numpy.array([-110, -110, -110, -110, -110])
        self.mock_fix.get_bbox.return_value = [-114, -104, 31.5, 37.5]

    def test_plot_catalog_default(self):
        # Test plot with default settings4
        ax = plot_catalog(self.mock_catalog, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def test_plot_catalog_title(self):
        # Test plot with default settings
        ax = plot_catalog(self.mock_catalog, show=show_plots, title=self.mock_catalog.name)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "Mock Catalog")

    def test_plot_catalog_without_legend(self):
        # Test plot with legend
        ax = plot_catalog(self.mock_catalog, mag_scale=7, show=show_plots, legend=False)
        legend = ax.get_legend()
        self.assertIsNone(legend)

    def test_plot_catalog_custom_legend(self):

        ax = plot_catalog(self.mock_catalog, mag_ticks=5, show=show_plots)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        mags = self.mock_catalog.get_magnitudes()
        mag_bins = numpy.linspace(min(mags), max(mags), 3, endpoint=True)
        ax = plot_catalog(self.mock_catalog, mag_ticks=mag_bins, show=show_plots)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

    def test_plot_catalog_correct_sizing(self):

        ax = plot_catalog(
            self.mock_fix,
            figsize=(4, 6),
            mag_ticks=[4, 5, 6, 7, 8],
            legend_loc="right",
            show=show_plots,
        )
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

    def test_plot_catalog_custom_sizes(self):

        ax = plot_catalog(self.mock_catalog, size=5, max_size=800, power=6, show=show_plots)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

    def test_plot_catalog_same_size(self):

        ax = plot_catalog(self.mock_catalog, size=30, power=0, show=show_plots)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

    def test_plot_catalog_with_custom_extent(self):
        # Test plot with custom extent
        custom_extent = (-130, 20, 10, 80)
        ax = plot_catalog(self.mock_catalog, extent=custom_extent, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)
        self.assertAlmostEqual(ax.get_extent(crs=ccrs.PlateCarree()), custom_extent)

    def test_plot_catalog_global(self):
        # Test plot with global extent
        ax = plot_catalog(self.mock_catalog, set_global=True, show=show_plots)
        self.assertTrue(ax.spines["geo"].get_visible())

    def test_plot_catalog_with_region_border(self):
        # Test plot with region border
        ax = plot_catalog(self.mock_catalog, show=show_plots, plot_region=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_catalog_with_no_grid(self):
        # Test plot with grid disabled
        ax = plot_catalog(self.mock_catalog, show=show_plots, grid=False)
        gl = ax.gridlines()
        self.assertIsNotNone(gl)

    def test_plot_catalog_w_basemap(self):
        # Test plot with default settings
        ax = plot_catalog(self.mock_catalog, basemap="stock_img", show=show_plots)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def test_plot_catalog_w_basemap_stream_kwargs(self):

        projection = ccrs.Mercator()
        ax = plot_catalog(
            self.mock_catalog,
            basemap=None,
            projection=projection,
            coastline=True,
            borders=True,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def test_plot_catalog_w_approx_projection(self):
        projection = "approx"
        ax = plot_catalog(
            self.mock_catalog,
            basemap="stock_img",
            projection=projection,
            coastline=True,
            borders=True,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestPlotSpatialDataset(TestPlots):

    def setUp(self):
        # Mock region and data
        self.region = self.MockRegion()
        self.gridded_data = numpy.random.rand(len(self.region.ys), len(self.region.xs))

    class MockRegion:
        def __init__(self):
            self.xs = numpy.linspace(-20, 20, 100)
            self.ys = numpy.linspace(-10, 10, 50)

        @staticmethod
        def get_bbox():
            return [-20, 20, -10, 10]

        @staticmethod
        def tight_bbox():
            return numpy.array([[-20, -10], [20, -10], [20, 10], [-20, 10], [-20, -10]])

    def test_default_plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(self.gridded_data, self.region, ax=ax)
        self.assertIsInstance(ax, plt.Axes)

    def test_extent_setting_w_ax(self):
        extent = (-30, 30, -20, 20)
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, extent=extent, show=show_plots
        )
        numpy.testing.assert_array_almost_equal(ax.get_extent(crs=ccrs.PlateCarree()), extent)

    def test_extent_setting(self):
        extent = (-30, 30, -20, 20)
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, extent=extent, show=show_plots
        )
        numpy.testing.assert_array_almost_equal(ax.get_extent(crs=ccrs.PlateCarree()), extent)
        # self.assertAlmostEqual(ax.get_extent(crs=ccrs.PlateCarree()), extent)

    def test_color_mapping(self):
        cmap = plt.get_cmap("plasma")
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, ax=ax, colormap=cmap, show=show_plots
        )
        self.assertIsInstance(ax.collections[0].cmap, colors.ListedColormap)

    def test_gridlines(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, ax=ax, grid=True, show=show_plots
        )
        self.assertTrue(ax.gridlines())

    def test_alpha_transparency(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, ax=ax, alpha=0.5, show=show_plots
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_alpha_exp(self):
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, alpha_exp=0.5, include_cbar=True, show=show_plots
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_include_colorbar(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, ax=ax, include_cbar=True, show=show_plots
        )
        colorbars = [
            child
            for child in ax.get_figure().get_children()
            if isinstance(child, plt.Axes) and "Colorbar" in child.get_label()
        ]
        self.assertGreater(len(colorbars), 0)

    def test_no_region_border(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax = plot_gridded_dataset(
            self.gridded_data, self.region, ax=ax, plot_region=False, show=show_plots
        )
        lines = ax.get_lines()
        self.assertEqual(len(lines), 0)

    def test_plot_spatial_dataset_w_basemap_stream_kwargs(self):

        projection = ccrs.Mercator()
        ax = plot_gridded_dataset(
            self.gridded_data,
            self.region,
            extent=[-20, 40, -5, 25],
            projection=projection,
            coastline=True,
            borders=True,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
            plot_region=False,
        )

        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def test_plot_spatial_dataset_w_approx_projection(self):
        projection = "approx"
        ax = plot_gridded_dataset(
            self.gridded_data,
            self.region,
            basemap="stock_img",
            extent=[-20, 40, -5, 25],
            projection=projection,
            coastline=True,
            borders=True,
            grid=True,
            grid_labels=True,
            grid_fontsize=8,
            show=show_plots,
            plot_region=False,
        )

        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), "")

    def tearDown(self):
        plt.close("all")
        del self.region
        del self.gridded_data
        gc.collect()


class TestHelperFunctions(TestPlots):

    def setUp(self):
        # Set up a mock catalog with basic properties
        self.mock_catalog = MagicMock()
        self.mock_catalog.get_bbox.return_value = [-120, -115, 30, 35]
        self.mock_catalog.get_magnitudes.return_value = numpy.array([3.5, 4.0, 4.5])

        # Mock region if needed
        self.mock_catalog.region.get_bbox.return_value = [-120, -115, 30, 35]

    def test_get_marker_style(self):
        self.assertEqual(_get_marker_style(1, [2, 3], False), "ro")
        self.assertEqual(_get_marker_style(2, [1, 3], False), "gs")
        self.assertEqual(_get_marker_style(1, [2, 3], True), "ro")
        self.assertEqual(_get_marker_style(4, [2, 3], True), "gs")

    def test_get_marker_t_color(self):
        self.assertEqual(_get_marker_t_color([1, 2]), "green")
        self.assertEqual(_get_marker_t_color([-2, -1]), "red")
        self.assertEqual(_get_marker_t_color([-1, 1]), "grey")

    def test_get_marker_w_color(self):
        self.assertTrue(_get_marker_w_color(0.01, 95))
        self.assertFalse(_get_marker_w_color(0.99, 95))

    def test_get_axis_limits(self):
        pnts = numpy.array([1, 2, 3, 4, 5])
        expected_limits = (0.8, 5.2)
        self.assertEqual(_get_axis_limits(pnts, border=0.05), expected_limits)

    def test_autosize_scatter(self):
        values = numpy.array([1, 2, 3, 4, 5])
        expected_sizes = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        result = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

        values = numpy.array([1, 2, 3, 4, 5])
        min_val = 0
        max_val = 10
        expected_sizes = _autosize_scatter(
            values, min_size=50.0, max_size=400.0, power=3.0, min_val=min_val, max_val=max_val
        )
        result = _autosize_scatter(
            values, min_size=50.0, max_size=400.0, power=3.0, min_val=min_val, max_val=max_val
        )
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

        values = numpy.array([1, 2, 3, 4, 5])
        power = 2.0
        expected_sizes = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=power)
        result = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=power)
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

        values = numpy.array([1, 2, 3, 4, 5])
        power = 0.0
        expected_sizes = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=power)
        result = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=power)
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

        values = numpy.array([5, 5, 5, 5, 5])
        expected_sizes = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        result = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

        values = numpy.array([10, 100, 1000, 10000, 100000])
        expected_sizes = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        result = _autosize_scatter(values, min_size=50.0, max_size=400.0, power=3.0)
        numpy.testing.assert_almost_equal(result, expected_sizes, decimal=2)

    def test_autoscale_histogram(self):
        fig, ax = plt.subplots()
        simulated = numpy.random.normal(size=1000)
        observation = numpy.random.normal(size=1000)
        bin_edges = numpy.linspace(-5, 5, 21)

        ax = _autoscale_histogram(ax, bin_edges, simulated, observation)

        x_min, x_max = ax.get_xlim()

        self.assertGreaterEqual(x_min, -6)
        self.assertLessEqual(x_max, 6)
        self.assertGreater(x_max, x_min)

    def test_annotate_distribution_plot(self):
        # Mock evaluation_result for Catalog N-Test
        evaluation_result = Mock()
        evaluation_result.name = "Catalog N-Test"
        evaluation_result.sim_name = "Simulated Catalog"
        evaluation_result.quantile = [0.25, 0.75]
        evaluation_result.observed_statistic = 5.0

        ax = plt.gca()
        plot_args = {
            "annotation_text": None,
            "annotation_xy": (0.5, 0.5),
            "annotation_fontsize": 12,
            "xlabel": None,
            "ylabel": None,
            "title": None,
        }

        ax = _annotate_distribution_plot(
            ax, evaluation_result, auto_annotate=True, plot_args=plot_args
        )

        # Assertions to check if the annotations were correctly set
        self.assertEqual(ax.get_xlabel(), "Event Count")
        self.assertEqual(ax.get_ylabel(), "Number of Catalogs")
        self.assertEqual(ax.get_title(), "Catalog N-Test: Simulated Catalog")

        annotation = ax.texts[0].get_text()
        expected_annotation = (
            f"$\\delta_1 = P(X \\geq x) = 0.25$\n"
            f"$\\delta_2 = P(X \\leq x) = 0.75$\n"
            f"$\\omega = 5.00$"
        )
        self.assertEqual(annotation, expected_annotation)

    def test_calculate_spatial_extent(self):
        # Test with plot_region and set_global=False
        extent = _calculate_spatial_extent(
            self.mock_catalog, set_global=False, region_border=True
        )
        expected_extent = [-120.25, -114.75, 29.75, 35.25]
        self.assertEqual(extent, expected_extent)

        # Test with set_global=True
        extent = _calculate_spatial_extent(
            self.mock_catalog, set_global=True, region_border=True
        )
        self.assertIsNone(extent)

        # Test with no plot_region
        extent = _calculate_spatial_extent(
            self.mock_catalog, set_global=False, region_border=False
        )
        self.assertEqual(extent, expected_extent)

    def test_create_geo_axes(self):
        # Test GeoAxes creation with no extent (global)
        ax = _create_geo_axes(
            figsize=(10, 8), extent=None, projection=ccrs.PlateCarree(), set_global=True
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertAlmostEqual(ax.get_xlim(), (-180, 180))
        self.assertAlmostEqual(ax.get_ylim(), (-90, 90))

        # Test GeoAxes creation with a specific extent
        extent = (-125, -110, 25, 40)
        ax = _create_geo_axes(
            figsize=(10, 8), extent=extent, projection=ccrs.PlateCarree(), set_global=False
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertAlmostEqual(ax.get_extent(), extent)

    def test_add_gridlines(self):
        # Test adding gridlines to an axis
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        _add_gridlines(ax, grid_labels=True, grid_fontsize=12)
        gl = ax.gridlines()
        self.assertIsNotNone(gl)

    @patch("csep.utils.plots.img_tiles.GoogleTiles")
    def test_get_basemap_google_satellite(self, mock_google_tiles):
        # Simulate return value for Google satellite
        mock_google_tiles.return_value = MagicMock()
        tiles = _get_basemap("google-satellite")
        mock_google_tiles.assert_called_once_with(style="satellite", cache=True)
        self.assertIsNotNone(tiles)

    @patch("csep.utils.plots.img_tiles.GoogleTiles")
    def test_get_basemap_esri_terrain(self, mock_google_tiles):
        # Simulate return value for ESRI terrain
        mock_google_tiles.return_value = MagicMock()
        tiles = _get_basemap("ESRI_terrain")
        mock_google_tiles.assert_called_once_with(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/"
            "MapServer/tile/{z}/{y}/{x}.jpg",
            cache=True,
        )
        self.assertIsNotNone(tiles)

    @patch("csep.utils.plots.img_tiles.GoogleTiles")
    def test_get_basemap_custom_url(self, mock_google_tiles):
        # Simulate return value for custom URL
        custom_url = "https://custom.tileserver.com/tiles/{z}/{y}/{x}.jpg"
        mock_google_tiles.return_value = MagicMock()
        tiles = _get_basemap(custom_url)
        mock_google_tiles.assert_called_once_with(url=custom_url, cache=True)
        self.assertIsNotNone(tiles)

    def test_plot_basemap_basic(self):
        basemap = "stock_img"
        extent = [-180, 180, -90, 90]
        ax = plot_basemap(basemap, extent, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_basemap_no_basemap(self):
        # Test with no basemap (should handle it gracefully)
        extent = [-75, -71, -44, -40]
        ax = plot_basemap(None, extent, show=show_plots)

        # Assertions
        self.assertIsInstance(ax, plt.Axes)

    def test_default_colormap(self):
        cmap, alpha = _get_colormap("viridis", 0)
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        expected_cmap = plt.get_cmap("viridis")
        self.assertTrue(numpy.allclose(cmap.colors, expected_cmap(numpy.arange(cmap.N))))

    def test_custom_colormap(self):
        cmap = plt.get_cmap("plasma")
        cmap, alpha = _get_colormap(cmap, 0)
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        expected_cmap = plt.get_cmap("plasma")
        self.assertTrue(numpy.allclose(cmap.colors, expected_cmap(numpy.arange(cmap.N))))

    def test_alpha_exponent(self):
        cmap, alpha = _get_colormap("viridis", 0.5)
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsNone(alpha)
        # Check that alpha values are correctly modified
        self.assertTrue(numpy.all(cmap.colors[:, -1] == numpy.linspace(0, 1, cmap.N) ** 0.5))

    def test_no_alpha_exponent(self):
        cmap, alpha = _get_colormap("viridis", 0)
        self.assertEqual(alpha, 1)
        self.assertTrue(numpy.all(cmap.colors[:, -1] == 1))  # No alpha modification

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestPlotAlarmBasedEvaluations(unittest.TestCase):

    def setUp(self):
        # Set up a mock catalog with basic properties
        self.forecast = MagicMock()
        self.forecast.region = MagicMock()
        self.forecast.spatial_counts.return_value = numpy.array([1, 10, 2, 40, 50, 2, 70, 80])
        self.forecast.name = "Test Forecast"

        self.catalog = MagicMock()
        self.catalog.region = self.forecast.region
        self.catalog.spatial_counts.return_value = numpy.array([2, 8, 0, 38, 52, 0, 67, 78])
        self.catalog.region.get_cell_area.return_value = numpy.array([1, 1, 1, 1, 2, 2, 2, 2])
        self.catalog.name = "Test Catalog"

    def test_plot_concentration_ROC_diagram(self):
        ax = plot_concentration_ROC_diagram(self.forecast, self.catalog, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_ROC_diagram(self):
        ax = plot_ROC_diagram(self.forecast, self.catalog, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_Molchan_diagram(self):

        ax = plot_Molchan_diagram(self.forecast, self.catalog, show=show_plots)
        self.assertIsInstance(ax, plt.Axes)

    def tearDown(self):
        plt.close("all")
        gc.collect()


class TestProcessDistribution(unittest.TestCase):

    def setUp(self):
        self.result_poisson = mock.Mock()
        self.result_poisson.test_distribution = ["poisson", 10]
        self.result_poisson.observed_statistic = 8

        self.result_neg_binom = mock.Mock()
        self.result_neg_binom.test_distribution = ["negative_binomial", 10, 12]
        self.result_neg_binom.observed_statistic = 8

        self.result_empirical = mock.Mock()
        self.result_empirical.test_distribution = numpy.random.normal(10, 2, 100)
        self.result_empirical.observed_statistic = 8

    def test_process_distribution_poisson(self):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            self.result_poisson,
            percentile=95,
            normalize=False,
            one_sided_lower=False,
        )
        self.assertAlmostEqual(mean, 10)
        self.assertAlmostEqual(observed_statistic, 8)
        self.assertTrue(plow < mean < phigh)

    def test_process_distribution_negative_binomial(self):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            self.result_neg_binom,
            percentile=95,
            normalize=False,
            one_sided_lower=False,
        )
        self.assertAlmostEqual(mean, 10)
        self.assertAlmostEqual(observed_statistic, 8)
        self.assertTrue(plow < mean < phigh)

    def test_process_distribution_empirical(self):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            self.result_empirical,
            percentile=95,
            normalize=False,
            one_sided_lower=False,
        )
        self.assertAlmostEqual(mean, numpy.mean(self.result_empirical.test_distribution))
        self.assertAlmostEqual(observed_statistic, 8)
        self.assertTrue(plow < mean < phigh)

    def test_process_distribution_empirical_normalized(self):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            self.result_empirical,
            percentile=95,
            normalize=True,
            one_sided_lower=False,
        )
        self.assertAlmostEqual(
            mean,
            numpy.mean(
                self.result_empirical.test_distribution
                - self.result_empirical.observed_statistic
            ),
        )
        self.assertAlmostEqual(observed_statistic, 0)
        self.assertTrue(plow < mean < phigh)

    def test_process_distribution_empirical_one_sided(self):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            self.result_empirical,
            percentile=95,
            normalize=False,
            one_sided_lower=True,
        )
        self.assertAlmostEqual(mean, numpy.mean(self.result_empirical.test_distribution))
        self.assertAlmostEqual(observed_statistic, 8)
        self.assertTrue(plow <= mean <= phigh)


if __name__ == "__main__":
    unittest.main()
