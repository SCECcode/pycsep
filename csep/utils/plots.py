import os
import shutil
import string
import warnings
from typing import TYPE_CHECKING, Optional, Any, List, Union, Tuple

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as pyplot
# Third-party imports
import numpy
import numpy as np
import pandas as pandas
import rasterio
import scipy.stats
from cartopy.io import img_tiles
from cartopy.io.img_tiles import GoogleWTS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.lines import Line2D
from rasterio import DatasetReader
from rasterio import plot as rioplot
from scipy.integrate import cumulative_trapezoid

# PyCSEP imports
import csep.utils.time_utils
from csep.utils.calc import bin1d_vec
from csep.utils.constants import SECONDS_PER_DAY, CSEP_MW_BINS, SECONDS_PER_HOUR
from csep.utils.time_utils import datetime_to_utc_epoch

if TYPE_CHECKING:
    from csep.core.catalogs import CSEPCatalog
    from csep.core.forecasts import GriddedForecast, CatalogForecast
    from csep.models import EvaluationResult
    from csep.core.regions import CartesianGrid2D


DEFAULT_PLOT_ARGS = {
    # General figure/axes handling
    "figsize": None,
    "tight_layout": True,
    "grid": True,
    "title": None,
    "title_fontsize": 16,
    "xlabel": None,
    "ylabel": None,
    "xlabel_fontsize": 12,
    "ylabel_fontsize": 12,
    "xticks_fontsize": 12,
    "yticks_fontsize": 12,
    "xlim": None,
    "ylim": None,
    "legend": True,
    "legend_loc": "best",
    "legend_fontsize": 10,
    "legend_title": None,
    "legend_titlesize": None,
    "legend_labelspacing": 1,
    "legend_borderpad": 0.4,
    "legend_framealpha": None,
    # Line/Scatter parameters
    "color": "steelblue",
    "alpha": 0.8,
    "linewidth": 1,
    "size": 5,
    "marker": "o",
    "markersize": 5,
    "markercolor": "steelblue",
    "markeredgecolor": "black",
    # Time-Series
    "datetime_locator": AutoDateLocator(),
    "datetime_formatter": DateFormatter("%Y-%m-%d"),
    # Consistency and Comparison tests
    "capsize": 2,
    "hbars": True,
    # Specific to spatial plotting
    "grid_labels": True,
    "grid_fontsize": 8,
    "region_color": "black",
    "coastline": True,
    "coastline_color": "black",
    "coastline_linewidth": 1.5,
    "borders": False,
    "borders_color": "black",
    "borders_linewidth": 1.5,
    # Color bars
    "colorbar_labelsize": 12,
    "colorbar_ticksize": 10,
}


########################
# Data-exploratory plots
########################
def plot_magnitude_vs_time(
    catalog: "CSEPCatalog",
    ax: Optional[Axes] = None,
    color: Optional[str] = "steelblue",
    size: Optional[int] = 4,
    mag_scale: Optional[int] = 6,
    alpha: Optional[float] = 0.5,
    show: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Plots magnitude versus time for a given catalog.

    Args:
        catalog (:class:`csep.core.catalogs.CSEPCatalog`):
            Catalog of seismic events to be plotted.
        ax (Axes):
            Matplotlib axis object on which to plot. If not provided, a new figure and axis are
            created.
        color (str):
            Color of the scatter plot points. If not provided, defaults to value in
            `DEFAULT_PLOT_ARGS`.
        size (int):
            Size of the scatter plot markers.
        mag_scale (int):
            Scaling factor for the magnitudes.
        alpha (float):
            Transparency level for the scatter plot points. If not provided, defaults to value
            in `DEFAULT_PLOT_ARGS`.
        show (bool):
            Whether to display the plot. Defaults to `False`.
        **kwargs:
            Additional keyword arguments for customizing the plot. These are merged with
            `DEFAULT_PLOT_ARGS`.


    Returns:
        Axes:
            The Matplotlib axes object with the plotted data.

    """

    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Plot data
    mag = catalog.data["magnitude"]
    datetimes = catalog.get_datetimes()
    ax.scatter(
        datetimes,
        mag,
        marker="o",
        c=color,
        s=_autosize_scatter(size, mag, mag_scale),
        alpha=alpha,
    )

    # Set labels and title
    ax.set_xlabel(plot_args["xlabel"] or "Datetime", fontsize=plot_args["xlabel_fontsize"])
    ax.set_ylabel(plot_args["ylabel"] or "$M$", fontsize=plot_args["ylabel_fontsize"])
    ax.set_title(
        plot_args["title"] or "Magnitude vs. Time", fontsize=plot_args["title_fontsize"]
    )

    # Autoformat ticks and labels
    ax.xaxis.set_major_locator(plot_args["datetime_locator"])
    ax.xaxis.set_major_formatter(plot_args["datetime_formatter"])
    ax.grid(plot_args["grid"])
    fig.autofmt_xdate()

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


def plot_cumulative_events_versus_time(
    catalog_forecast: "CatalogForecast",
    observation: "CSEPCatalog",
    time_axis: str = "datetime",
    bins: int = 50,
    ax: Optional[matplotlib.axes.Axes] = None,
    sim_label: Optional[str] = "Simulated",
    obs_label: Optional[str] = "Observation",
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots the cumulative number of forecasted events versus observed events over time.

    Args:
        catalog_forecast (GriddedForecast): The forecasted catalogs.
        observation (CSEPCatalog): The observed catalog of events.
        time_axis (str): The type of time axis ('datetime', 'days', 'hours'). Defaults
            to 'datetime'.
        ax (Optional[pyplot.Axes]): The axes to plot on. If None, a new figure and
            axes are created.
        bins (int): The number of bins for time slicing. Defaults to 50.
        sim_label (str): Label for simulated data. Defaults to 'Simulated'.
        obs_label (str): Label for observed data. Defaults to 'Observation'.
        show (bool): If True, displays the plot. Defaults to False.
        **kwargs: Additional plotting arguments to override `DEFAULT_PLOT_ARGS`.

    Returns:
        pyplot.Axes: The axes with the cumulative event plot.
    """

    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Get information from stochastic event set
    if catalog_forecast.start_time is None or catalog_forecast.end_time is None:
        extreme_times = []
        for ses in catalog_forecast:
            if ses.start_time and ses.end_time:
                extreme_times.append(
                    (datetime_to_utc_epoch(ses.start_time), datetime_to_utc_epoch(ses.end_time))
                )
        start_time = numpy.min(numpy.array(extreme_times))
        end_time = numpy.max(numpy.array(extreme_times))
    else:
        start_time = datetime_to_utc_epoch(catalog_forecast.start_time)
        end_time = datetime_to_utc_epoch(catalog_forecast.end_time)

    # offsets to start at 0 time and converts from millis to hours
    time_bins, dt = numpy.linspace(start_time, end_time, bins, endpoint=True, retstep=True)
    n_bins = time_bins.shape[0]

    # Initialize an empty list to store binned counts for each stochastic event set
    binned_counts_list = []

    for i, ses in enumerate(catalog_forecast):
        n_events = ses.data.shape[0]
        ses_origin_time = ses.get_epoch_times()
        inds = bin1d_vec(ses_origin_time, time_bins)

        # Create a temporary array for the current stochastic event set
        temp_binned_counts = numpy.zeros(n_bins)
        for j in range(n_events):
            temp_binned_counts[inds[j]] += 1

        # Append the temporary array to the list
        binned_counts_list.append(temp_binned_counts)

    # Convert the list of arrays to a 2D NumPy array
    binned_counts = numpy.array(binned_counts_list)
    # Compute the cumulative sum along the specified axis
    summed_counts = numpy.cumsum(binned_counts, axis=1)

    # compute summary statistics for plotting
    fifth_per = numpy.percentile(summed_counts, 5, axis=0)
    first_quar = numpy.percentile(summed_counts, 25, axis=0)
    med_counts = numpy.percentile(summed_counts, 50, axis=0)
    second_quar = numpy.percentile(summed_counts, 75, axis=0)
    nine_fifth = numpy.percentile(summed_counts, 95, axis=0)

    # compute median for comcat data
    obs_binned_counts = numpy.zeros(n_bins)
    inds = bin1d_vec(observation.get_epoch_times(), time_bins)
    for j in range(observation.event_count):
        obs_binned_counts[inds[j]] += 1
    obs_summed_counts = numpy.cumsum(obs_binned_counts)

    # make all arrays start at zero
    time_bins = numpy.insert(time_bins, 0, 2 * time_bins[0] - time_bins[1])  # One DT before
    fifth_per = numpy.insert(fifth_per, 0, 0)
    first_quar = numpy.insert(first_quar, 0, 0)
    med_counts = numpy.insert(med_counts, 0, 0)
    second_quar = numpy.insert(second_quar, 0, 0)
    nine_fifth = numpy.insert(nine_fifth, 0, 0)
    obs_summed_counts = numpy.insert(obs_summed_counts, 0, 0)

    if time_axis == "datetime":
        time_bins = [csep.epoch_time_to_utc_datetime(i) for i in time_bins]
        ax.xaxis.set_major_locator(plot_args["datetime_locator"])
        ax.xaxis.set_major_formatter(plot_args["datetime_formatter"])
        ax.set_xlabel(plot_args["xlabel"] or "Datetime", fontsize=plot_args["xlabel_fontsize"])
        fig.autofmt_xdate()
    elif time_axis == "days":
        time_bins = (time_bins - time_bins[0]) / (SECONDS_PER_DAY * 1000)
        ax.set_xlabel(
            plot_args["xlabel"] or "Days after Mainshock", fontsize=plot_args["xlabel_fontsize"]
        )
    elif time_axis == "hours":
        time_bins = (time_bins - time_bins[0]) / (SECONDS_PER_HOUR * 1000)
        ax.set_xlabel(
            plot_args["xlabel"] or "Hours after Mainshock",
            fontsize=plot_args["xlabel_fontsize"],
        )

    # Plotting
    ax.plot(
        time_bins,
        obs_summed_counts,
        color="black",
        linewidth=plot_args["linewidth"],
        label=obs_label,
    )
    ax.plot(
        time_bins,
        med_counts,
        color=plot_args["color"],
        linewidth=plot_args["linewidth"],
        label=sim_label,
    )
    ax.fill_between(
        time_bins, fifth_per, nine_fifth, color=plot_args["color"], alpha=0.2, label="5%-95%"
    )
    ax.fill_between(
        time_bins, first_quar, second_quar, color=plot_args["color"], alpha=0.5, label="25%-75%"
    )

    # Plot formatting
    ax.grid(plot_args["grid"])
    ax.set_ylabel(
        plot_args["ylabel"] or "Cumulative event counts", fontsize=plot_args["ylabel_fontsize"]
    )
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])

    if plot_args["legend"]:
        ax.legend(loc=plot_args["legend_loc"], fontsize=plot_args["legend_fontsize"])
    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()
    return ax


def plot_magnitude_histogram(
    catalog_forecast: Union["CatalogForecast", List["CSEPCatalog"]],
    observation: "CSEPCatalog",
    magnitude_bins: Optional[Union[List[float], numpy.ndarray]] = None,
    percentile: int = 95,
    ax: Optional["matplotlib.axes.Axes"] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Generates a semi-log magnitude histogram comparing a catalog-based forecast with observed
    data. The forecast's median and uncertainty intervals are displayed along with the observed
    event counts.

    Args:
        catalog_forecast (CatalogForecast, List[CSEPCatalog]): A Catalog-based forecast
        observation (CSEPCatalog): The observed catalog for comparison.
        magnitude_bins (Optional[Union[List[float], numpy.ndarray]]): The bins for magnitude
            histograms. If None, defaults to the region magnitudes or standard CSEP bins.
        percentile (int): The percentile used for uncertainty intervals (default: 95).
        ax (Optional[pyplot.Axes]): The axes object to draw the plot on. If None, a new
            figure and axes are created.
        show (bool): Whether to display the plot immediately (default: False).
        **kwargs: Additional keyword arguments for plot customization. These override defaults.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Get magnitudes from observations and (lazily) from forecast
    forecast_mws = list(map(lambda x: x.get_magnitudes(), catalog_forecast))
    obs_mw = observation.get_magnitudes()
    n_obs = observation.get_number_of_events()

    # Get magnitude bins from args, if not from region, or lastly from standard CSEP bins.
    if magnitude_bins is None:
        try:
            magnitude_bins = observation.region.magnitudes
        except AttributeError:
            magnitude_bins = CSEP_MW_BINS

    def get_histogram_synthetic_cat(x, mags, normed=True):
        n_temp = len(x)
        if normed and n_temp != 0:
            temp_scale = n_obs / n_temp
            hist = numpy.histogram(x, bins=mags)[0] * temp_scale
        else:
            hist = numpy.histogram(x, bins=mags)[0]
        return hist

    # get histogram values
    forecast_hist = numpy.array(
        list(map(lambda x: get_histogram_synthetic_cat(x, magnitude_bins), forecast_mws))
    )

    obs_hist, bin_edges = numpy.histogram(obs_mw, bins=magnitude_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Compute statistics for the forecast histograms
    forecast_median = numpy.median(forecast_hist, axis=0)
    forecast_low = numpy.percentile(forecast_hist, (100 - percentile) / 2.0, axis=0)
    forecast_high = numpy.percentile(forecast_hist, 100 - (100 - percentile) / 2.0, axis=0)

    forecast_err_lower = forecast_median - forecast_low
    forecast_err_upper = forecast_high - forecast_median

    # Plot observed histogram
    ax.semilogy(
        bin_centers,
        obs_hist,
        color=plot_args["color"],
        marker="o",
        lw=0,
        markersize=plot_args["markersize"],
        label="Observation",
        zorder=3,
    )

    # Plot forecast histograms as bar plot with error bars
    ax.plot(
        bin_centers,
        forecast_median,
        ".",
        markersize=plot_args["markersize"],
        color="darkred",
        label="Forecast Median",
    )
    ax.errorbar(
        bin_centers,
        forecast_median,
        yerr=[forecast_err_lower, forecast_err_upper],
        fmt="None",
        color="darkred",
        markersize=plot_args["markersize"],
        capsize=plot_args["capsize"],
        linewidth=plot_args["linewidth"],
        label="Forecast (95% CI)",
    )

    # Scale x-axis
    if plot_args["xlim"]:
        ax.set_xlim(plot_args["xlim"])
    else:
        ax = _autoscale_histogram(
            ax, magnitude_bins, numpy.hstack(forecast_mws), obs_mw, mass=100
        )

    # Format plot
    ax.grid(plot_args["grid"])
    ax.legend(loc=plot_args["legend_loc"], fontsize=plot_args["legend_fontsize"])
    ax.set_xlabel(plot_args["xlabel"] or "Magnitude", fontsize=plot_args["xlabel_fontsize"])
    ax.set_ylabel(plot_args["ylabel"] or "Event count", fontsize=plot_args["ylabel_fontsize"])
    ax.set_title(
        plot_args["title"] or "Magnitude Histogram", fontsize=plot_args["title_fontsize"]
    )

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


#####################
# Single Result plots
#####################
def plot_distribution_test(
    evaluation_result: "EvaluationResult",
    bins: Union[str, int, List[Any]] = "fd",
    percentile: Optional[int] = 95,
    ax: Optional[matplotlib.axes.Axes] = None,
    auto_annotate: Union[bool, dict] = True,
    sim_label: str = "Simulated",
    obs_label: str = "Observation",
    legend: bool = True,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a histogram of a single statistic for stochastic event sets and observations.

    Args:
        evaluation_result (EvaluationResult): Object containing test distributions and
            observed statistics.
        bins (Union[str, int], optional): Binning strategy for the histogram. Defaults
            to 'fd'.
        percentile (int, optional): Percentile for shading regions. Defaults to None
            (use global setting).
        ax (Optional[matplotlib.axes.Axes], optional): Axes object to plot on. If None,
            creates a new figure and axes.
        auto_annotate (bool, dict): If True, automatically format the plot details based
            on the evaluation result. It can be customized by passing the keyword arguments
            `xlabel`, `ylabel`, `annotation_text`, `annotation_xy` and `annotation_fontsize`.
        sim_label (str, optional): Label for the simulated data.
        obs_label (str, optional): Label for the observation data.
        legend (Optional[bool], optional): Whether to display the legend. Defaults to
            global setting.
        show (bool, optional): If True, show the plot. Defaults to False.
        **kwargs: Additional keyword arguments for plot customization.

    Returns:
        matplotlib.axes.Axes: Matplotlib axes handle.
    """

    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Get distributions
    simulated = evaluation_result.test_distribution
    observation = evaluation_result.observed_statistic

    # Remove any potential nans from arrays
    simulated = numpy.array(simulated)
    simulated = simulated[~numpy.isnan(simulated)]

    # Plot forecast statistic histogram
    n, bin_edges, patches = ax.hist(
        simulated,
        bins=bins,
        label=sim_label,
        color=plot_args["color"],
        alpha=plot_args["alpha"],
        edgecolor=None,
        linewidth=0,
    )

    # Plot observation statistic value/distribution
    if isinstance(observation, (float, int)):
        ax.axvline(
            x=observation,
            color="black",
            linestyle="--",
            label=obs_label + numpy.isinf(observation) * " (-inf)",
        )
    else:
        observation = observation[~numpy.isnan(observation)]
        ax.hist(
            observation,
            bins=bins,
            label=obs_label,
            edgecolor=None,
            linewidth=0,
            alpha=plot_args["alpha"],
            color="green",
        )

    # Annotate statistic analysis
    ax = _annotate_distribution_plot(ax, evaluation_result, auto_annotate, plot_args)

    # Format axis object
    if plot_args["xlim"] is None:
        ax = _autoscale_histogram(ax, bin_edges, simulated, observation)
    else:
        ax.set_xlim(plot_args["xlim"])
    ax.grid(plot_args["grid"])

    if legend:
        ax.legend(loc=plot_args["legend_loc"], fontsize=plot_args["legend_fontsize"])

    # Color bars for rejection area (after setting legend)
    if percentile is not None:
        inc = (100 - percentile) / 2
        inc_high = 100 - inc
        inc_low = inc

        p_high = numpy.percentile(simulated, inc_high)
        idx_high = numpy.digitize(p_high, bin_edges)
        p_low = numpy.percentile(simulated, inc_low)
        idx_low = numpy.digitize(p_low, bin_edges)
        for idx in range(idx_low):
            patches[idx].set_fc("red")
        for idx in range(idx_high, len(patches)):
            patches[idx].set_fc("red")

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


def plot_calibration_test(
    evaluation_result: "EvaluationResult",
    percentile: float = 95,
    ax: Optional[matplotlib.axes.Axes] = None,
    label: Optional[str] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a calibration test (QQ plot) with confidence intervals.

    Args:
        evaluation_result (EvaluationResult): The evaluation result object containing the test distribution.
        percentile (float): Percentile to build confidence interval
        ax (Optional[matplotlib.axes.Axes]): Axes object to plot on. If None, creates a new figure.
        show (bool): If True, displays the plot. Default is False.
        label (Optional[str]): Label for the plotted data. If None, uses `evaluation_result.sim_name`.
        **kwargs: Additional keyword arguments for customizing the plot. These are merged with
            `DEFAULT_PLOT_ARGS`.

    Returns:
        pyplot.Axes: The matplotlib axes object containing the plot.
    """
    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Set up QQ plots and KS test
    n = len(evaluation_result.test_distribution)
    k = numpy.arange(1, n + 1)
    # Plotting points for uniform quantiles
    pp = k / (n + 1)
    # Compute confidence intervals for order statistics using beta distribution
    inf = (100 - percentile) / 2
    sup = 100 - (100 - percentile) / 2
    ulow = scipy.stats.beta.ppf(inf / 100, k, n - k + 1)
    uhigh = scipy.stats.beta.ppf(sup / 100, k, n - k + 1)

    # Quantiles should be sorted for plotting
    sorted_td = numpy.sort(evaluation_result.test_distribution)

    if ax is None:
        fig, ax = pyplot.subplots()
    else:
        ax = ax

    # Plot QQ plot
    ax.plot(
        sorted_td,
        pp,
        linewidth=0,
        label=label or evaluation_result.sim_name,
        c=plot_args["color"],
        marker=plot_args["marker"],
        markersize=plot_args["markersize"],
    )

    # Plot uncertainty on uniform quantiles
    ax.plot(pp, pp, "-k")
    ax.plot(ulow, pp, ":k")
    ax.plot(uhigh, pp, ":k")

    # Format plot
    ax.grid(plot_args["grid"])
    ax.set_title(plot_args["title"] or "Calibration test", fontsize=plot_args["title_fontsize"])
    ax.set_xlabel(
        plot_args["xlabel"] or "Quantile scores", fontsize=plot_args["xlabel_fontsize"]
    )
    ax.set_ylabel(
        plot_args["ylabel"] or "Standard uniform quantiles",
        fontsize=plot_args["ylabel_fontsize"],
    )
    ax.legend(loc=plot_args["legend_loc"], fontsize=plot_args["legend_fontsize"])

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


#####################
# Results batch plots
#####################
def plot_comparison_test(
    results_t: List["EvaluationResult"],
    results_w: Optional[List["EvaluationResult"]] = None,
    percentile: int = 95,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    **kwargs,
) -> pyplot.Axes:
    """
    Plots a list of T-Test (and optional W-Test) results on a single axis.

    Args:
        results_t (List[EvaluationResult]): List of T-Test results.
        results_w (Optional[List[EvaluationResult]]): List of W-Test results. If provided, they
             are plotted alongside the T-Test results.
        percentile (int): Percentile for coloring W-Test results. Default is 95.
        ax (Optional[matplotlib.axes.Axes]): Matplotlib axes object to plot on. If None, a new
            figure and axes are created.
        show (bool): If True, the plot is displayed after creation. Default is False.
        **kwargs: Additional plotting arguments to override defaults.

    Returns:
        pyplot.Axes: The matplotlib axes object containing the plot.
    """

    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Iterate through T-test results, or jointly through t- and w- test results
    Results = zip(results_t, results_w) if results_w else zip(results_t)
    for index, result in enumerate(Results):
        result_t = result[0]
        result_w = result[1] if results_w else None

        # Get confidence bounds
        ylow = result_t.observed_statistic - result_t.test_distribution[0]
        yhigh = result_t.test_distribution[1] - result_t.observed_statistic
        color = _get_marker_t_color(result_t.test_distribution)

        if not numpy.isinf(result_t.observed_statistic):
            # Plot observed statistic and confidence bounds
            ax.errorbar(
                index,
                result_t.observed_statistic,
                yerr=numpy.array([[ylow, yhigh]]).T,
                color=color,
                linewidth=plot_args["linewidth"],
                capsize=plot_args["capsize"],
            )

            facecolor = "white"
            if result_w is not None:
                if _get_marker_w_color(result_w.quantile, percentile):
                    facecolor = _get_marker_t_color(result_t.test_distribution)

            ax.plot(
                index,
                result_t.observed_statistic,
                marker="o",
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markersize=plot_args["markersize"],
            )
        else:
            print(
                f"Diverging information gain for forecast {result_t.sim_name}, index {index}. "
                f"Check for zero-valued bins within the forecast"
            )
            ax.axvspan(index - 0.5, index + 0.5, alpha=0.5, facecolor="red")

    # Format plot
    ax.axhline(y=0, linestyle="--", color="black")
    ax.set_xticks(numpy.arange(len(results_t)))
    ax.set_xticklabels(
        [res.sim_name[0] for res in results_t],
        rotation=90,
        fontsize=plot_args["xlabel_fontsize"],
    )
    ax.set_ylabel(
        plot_args["ylabel"] or "Information gain per earthquake",
        fontsize=plot_args["ylabel_fontsize"],
    )

    ax.set_title(plot_args["title"] or results_t[0].name)
    ax.set_ylim(plot_args["ylim"])
    ax.set_xlim([-0.5, len(results_t) - 0.5])

    if plot_args["grid"]:
        ax.yaxis.grid()
        ax.yaxis.set_major_locator(pyplot.MaxNLocator(integer=True))

    if plot_args["hbars"]:
        if len(results_t) > 2:
            ax.bar(
                ax.get_xticks(),
                numpy.array([9999] * len(ax.get_xticks())),
                bottom=-2000,
                width=(ax.get_xticks()[1] - ax.get_xticks()[0]),
                color=["gray", "w"],
                alpha=0.2,
            )

    if plot_args["legend"]:
        # Add custom legend to explain results
        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="T-test rejected"),
            Line2D([0], [0], color="green", lw=2, label="T-test non-rejected"),
            Line2D([0], [0], color="gray", lw=2, label="T-test indistinguishable"),
            Line2D(
                [0],
                [0],
                color="gray",
                lw=2,
                marker="o",
                markersize=6,
                markerfacecolor="green",
                label="W-test non-rejected",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                lw=2,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                label="W-test indistinguishable",
            ),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=plot_args["legend_fontsize"])

    if plot_args["tight_layout"]:
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax


def plot_consistency_test(
    eval_results: Union[List["EvaluationResult"], "EvaluationResult"],
    normalize: bool = False,
    one_sided_lower: bool = False,
    percentile: float = 95,
    variance: Optional[float] = None,
    ax: Optional[pyplot.Axes] = None,
    plot_mean: bool = False,
    color: str = "black",
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots the results from ultiple consistency tests. The distribution of score results from
    multiple realizations of a model are plotted as a line representing for a given percentile.
    The score of the observation under a model is plotted as a marker. The model is assumed
    inconsistent when the observation score lies outside from the model distribution for a
    two-sided test, or lies to the right of the distribution for a one-sided test.

    Args:
        eval_results (Union[List[EvaluationResult], EvaluationResult]):
            Evaluation results from one or multiple models
        normalize (bool):
            Normalize the forecast likelihood by observed likelihood (default: False).
        percentile (float):
            Percentile for the extent of the model score distribution (default: 95).
        one_sided_lower (bool):
            Plot for a one-sided test (default: False).
        variance (Optional[float]):
            Variance for negative binomial distribution (default: None).
        ax (Optional[pyplot.Axes]):
            Axes object to plot on (default: None).
        plot_mean (bool):
            Plot the mean of the test distribution (default: False).
        color (str):
            Color for the line representing a model score distribution (default: 'black').
        show (bool):
            If True, display the plot (default: False).
        **kwargs:
            Additional keyword arguments for plot customization from DEFAULT_PLOT_ARGS.

    Returns:
        matplotlib.axes.Axes: Matplotlib axes object with the consistency test plot.
    """

    # Initialize plot
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Ensure eval_results is a list
    results = list(eval_results) if isinstance(eval_results, list) else [eval_results]
    results.reverse()

    xlims = []

    for index, res in enumerate(results):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            res, percentile, variance, normalize, one_sided_lower
        )

        if not numpy.isinf(observed_statistic):  # Check if test result does not diverge
            percentile_lims = numpy.abs(numpy.array([[mean - plow, phigh - mean]]).T)
            ax.plot(
                observed_statistic,
                index,
                _get_marker_style(observed_statistic, (plow, phigh), one_sided_lower),
            )
            ax.errorbar(
                mean,
                index,
                xerr=percentile_lims,
                fmt="ko" if plot_mean else "k",
                capsize=plot_args["capsize"],
                linewidth=plot_args["linewidth"],
                ecolor=color,
            )
            # Determine the limits to use
            xlims.append((plow, phigh, observed_statistic))

            # Extend distribution to +inf, in case it is a one-sided test
            if one_sided_lower and observed_statistic >= plow and phigh < observed_statistic:
                xt = numpy.linspace(phigh, 99999, 100)
                yt = numpy.ones(100) * index
                ax.plot(xt, yt, linestyle="--", linewidth=plot_args["linewidth"], color=color)
        else:
            print(
                f"Observed statistic diverges for forecast {res.sim_name}, index {index}. "
                f"Check for zero-valued bins within the forecast"
            )
            ax.barh(index, 99999, left=-10000, height=1, color=["red"], alpha=0.5)

    # Plot formatting
    try:
        ax.set_xlim(*_get_axis_limits(xlims))
    except ValueError:
        raise ValueError("All EvaluationResults have infinite observed_statistics")
    ax.set_ylim([-0.5, len(results) - 0.5])
    ax.set_yticks(numpy.arange(len(results)))
    ax.set_yticklabels([res.sim_name for res in results], fontsize=plot_args["ylabel_fontsize"])
    ax.set_xlabel(
        plot_args["xlabel"] or "Statistic distribution", fontsize=plot_args["xlabel_fontsize"]
    )
    ax.tick_params(axis="x", labelsize=plot_args["xticks_fontsize"])
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])
    if plot_args["hbars"]:
        yTickPos = ax.get_yticks()
        if len(yTickPos) >= 2:
            ax.barh(
                yTickPos,
                numpy.array([99999] * len(yTickPos)),
                left=-10000,
                height=(yTickPos[1] - yTickPos[0]),
                color=["w", "gray"],
                alpha=0.2,
                zorder=0,
            )
    if plot_args["tight_layout"]:
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax


###################
# Alarm-based plots
###################
def plot_concentration_ROC_diagram(
    forecast,
    catalog,
    linear=True,
    axes=None,
    plot_uniform=True,
    show=True,
    figsize=(9, 8),
    forecast_linecolor="black",
    forecast_linestyle="-",
    observed_linecolor="blue",
    observed_linestyle="-",
    legend_fontsize=16,
    legend_loc="upper left",
    title_fontsize=18,
    label_fontsize=14,
    title="Concentration ROC Curve",
):
    if not catalog.region == forecast.region:
        raise RuntimeError("catalog region and forecast region must be identical.")

    name = forecast.name
    forecast_label = f"Forecast {name}" if name else "Forecast"
    observed_label = f"Observed {name}" if name else "Observed"

    # Initialize figure
    if axes is not None:
        ax = axes
    else:
        fig, ax = pyplot.subplots(figsize=figsize)

    area_km2 = catalog.region.get_cell_area()
    obs_counts = catalog.spatial_counts()
    rate = forecast.spatial_counts()

    I = numpy.argsort(rate)
    I = numpy.flip(I)

    fore_norm_sorted = numpy.cumsum(rate[I]) / numpy.sum(rate)
    area_norm_sorted = numpy.cumsum(area_km2[I]) / numpy.sum(area_km2)
    obs_norm_sorted = numpy.cumsum(obs_counts[I]) / numpy.sum(obs_counts)

    if plot_uniform:
        ax.plot(area_norm_sorted, area_norm_sorted, "k--", label="Uniform")

    ax.plot(
        area_norm_sorted,
        fore_norm_sorted,
        label=forecast_label,
        color=forecast_linecolor,
        linestyle=forecast_linestyle,
    )

    ax.step(
        area_norm_sorted,
        obs_norm_sorted,
        label=observed_label,
        color=observed_linecolor,
        linestyle=observed_linestyle,
    )

    ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
    ax.set_xlabel("False Positive Rate (Normalized Area)", fontsize=label_fontsize)

    if not linear:
        ax.set_xscale("log")

    ax.legend(loc=legend_loc, shadow=True, fontsize=legend_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    if show:
        pyplot.show()

    return ax


def plot_ROC_diagram(
    forecast,
    catalog,
    linear=True,
    axes=None,
    plot_uniform=True,
    show=True,
    figsize=(9, 8),
    forecast_linestyle="-",
    legend_fontsize=16,
    legend_loc="upper left",
    title_fontsize=16,
    label_fontsize=14,
    title="ROC Curve from contingency table",
):
    if not catalog.region == forecast.region:
        raise RuntimeError("catalog region and forecast region must be identical.")

    if axes is not None:
        ax = axes
    else:
        fig, ax = pyplot.subplots(figsize=figsize)

    rate = forecast.spatial_counts()
    obs_counts = catalog.spatial_counts()

    I = numpy.argsort(rate)
    I = numpy.flip(I)

    thresholds = (rate[I]) / numpy.sum(rate)
    obs_counts = obs_counts[I]

    Table_ROC = pandas.DataFrame({"Threshold": [], "H": [], "F": []})

    for threshold in thresholds:
        threshold = float(threshold)

        binary_forecast = numpy.where(thresholds >= threshold, 1, 0)
        forecastedYes_observedYes = obs_counts[(binary_forecast == 1) & (obs_counts > 0)]
        forecastedYes_observedNo = obs_counts[(binary_forecast == 1) & (obs_counts == 0)]
        forecastedNo_observedYes = obs_counts[(binary_forecast == 0) & (obs_counts > 0)]
        forecastedNo_observedNo = obs_counts[(binary_forecast == 0) & (obs_counts == 0)]

        H = len(forecastedYes_observedYes) / (
            len(forecastedYes_observedYes) + len(forecastedNo_observedYes)
        )
        F = len(forecastedYes_observedNo) / (
            len(forecastedYes_observedNo) + len(forecastedNo_observedNo)
        )

        threshold_row = {"Threshold": threshold, "H": H, "F": F}
        Table_ROC = pandas.concat(
            [Table_ROC, pandas.DataFrame([threshold_row])], ignore_index=True
        )

    Table_ROC = pandas.concat(
        [pandas.DataFrame({"H": [0], "F": [0]}), Table_ROC], ignore_index=True
    )

    ax.plot(
        Table_ROC["F"],
        Table_ROC["H"],
        label=forecast.name or "Forecast",
        color="black",
        linestyle=forecast_linestyle,
    )

    if plot_uniform:
        ax.plot(
            numpy.arange(0, 1.001, 0.001),
            numpy.arange(0, 1.001, 0.001),
            linestyle="--",
            color="gray",
            label="SUP",
        )

    ax.set_ylabel("Hit Rate", fontsize=label_fontsize)
    ax.set_xlabel("Fraction of false alarms", fontsize=label_fontsize)

    if not linear:
        ax.set_xscale("log")

    ax.set_yscale("linear")
    ax.tick_params(axis="x", labelsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)
    ax.legend(loc=legend_loc, shadow=True, fontsize=legend_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    if show:
        pyplot.show()

    return ax


def plot_Molchan_diagram(
    forecast,
    catalog,
    linear=True,
    axes=None,
    plot_uniform=True,
    show=True,
    figsize=(9, 8),
    forecast_linestyle="-",
    legend_fontsize=16,
    legend_loc="lower left",
    title_fontsize=16,
    label_fontsize=14,
    title="Molchan diagram",
    forecast_label=None,
    observed_label=None,
    color="black",
):
    """
    Plot the Molchan Diagram based on forecast and test catalogs using the contingency table.
    The Area Skill score and its error are shown in the legend.

    The Molchan diagram is computed following this procedure:
        (1) Obtain spatial rates from GriddedForecast and the observed events from the observed catalog.
        (2) Rank the rates in descending order (highest rates first).
        (3) Sort forecasted rates by ordering found in (2), and normalize rates so their sum is equal to unity.
        (4) Obtain binned spatial rates from the observed catalog.
        (5) Sort gridded observed rates by ordering found in (2).
        (6) Test each ordered and normalized forecasted rate defined in (3) as a threshold value to obtain the
            corresponding contingency table.
        (7) Define the "nu" (Miss rate) and "tau" (Fraction of spatial alarmed cells) for each threshold using the
            information provided by the corresponding contingency table defined in (6).

    Note that:
        (1) The testing catalog and forecast should have exactly the same time-window (duration).
        (2) Forecasts should be defined over the same region.
        (3) If calling this function multiple times, update the color in the arguments.
        (4) The user can choose the x-scale (linear or log).

    Args:
        forecast (:class: `csep.forecast.GriddedForecast`):
        catalog (:class:`AbstractBaseCatalog`): evaluation catalog
        linear (bool): if true, a linear x-axis is used; if false, a logarithmic x-axis is used.
        axes (:class:`matplotlib.pyplot.Axes`): Previously defined ax object.
        plot_uniform (bool): if true, include uniform forecast on plot.
        show (bool): if true, displays the plot.
        figsize (tuple): Figure size - default: (9, 8).
        forecast_linestyle (str): Linestyle for the forecast line - default: '-'.
        legend_fontsize (float): Fontsize of the plot legend - default: 16.
        legend_loc (str): Location of the plot legend - default: 'lower left'.
        title_fontsize (float): Fontsize of the plot title - default: 16.
        label_fontsize (float): Fontsize of the axis labels - default: 14.
        title (str): Title of the plot - default: 'Molchan diagram'.
        forecast_label (str): Label for the forecast in the legend - default: forecast name.
        observed_label (str): Label for the observed catalog in the legend - default: forecast name.
        color (str): Color of the forecast line - default: 'black'.

    Returns:
        :class:`matplotlib.pyplot.Axes` object

    Raises:
        TypeError: Throws error if a CatalogForecast-like object is provided.
        RuntimeError: Throws error if Catalog and Forecast do not have the same region.
    """
    if not catalog.region == forecast.region:
        raise RuntimeError("Catalog region and forecast region must be identical.")

    # Initialize figure
    if axes is not None:
        ax = axes
    else:
        fig, ax = pyplot.subplots(figsize=figsize)

    if forecast_label is None:
        forecast_label = forecast.name if forecast.name else ""

    if observed_label is None:
        observed_label = forecast.name if forecast.name else ""

    # Obtain forecast rates (or counts) and observed catalog aggregated in spatial cells
    rate = forecast.spatial_counts()
    obs_counts = catalog.spatial_counts()

    # Get index of rates (descending sort)
    I = numpy.argsort(rate)
    I = numpy.flip(I)

    # Order forecast and cells rates by highest rate cells first
    thresholds = (rate[I]) / numpy.sum(rate)
    obs_counts = obs_counts[I]

    Table_molchan = pandas.DataFrame(
        {
            "Threshold": [],
            "Successful_bins": [],
            "Obs_active_bins": [],
            "tau": [],
            "nu": [],
            "R_score": [],
        }
    )

    # Each forecasted and normalized rate is tested as a threshold value to define the
    # contingency table.
    for threshold in thresholds:
        threshold = float(threshold)

        binary_forecast = numpy.where(thresholds >= threshold, 1, 0)
        forecastedYes_observedYes = obs_counts[(binary_forecast == 1) & (obs_counts > 0)]
        forecastedYes_observedNo = obs_counts[(binary_forecast == 1) & (obs_counts == 0)]
        forecastedNo_observedYes = obs_counts[(binary_forecast == 0) & (obs_counts > 0)]
        forecastedNo_observedNo = obs_counts[(binary_forecast == 0) & (obs_counts == 0)]

        # Creating the DataFrame for the contingency table
        data = {
            "Observed": [len(forecastedYes_observedYes), len(forecastedNo_observedYes)],
            "Not Observed": [len(forecastedYes_observedNo), len(forecastedNo_observedNo)],
        }
        index = ["Forecasted", "Not Forecasted"]
        contingency_df = pandas.DataFrame(data, index=index)
        nu = contingency_df.loc["Not Forecasted", "Observed"] / contingency_df["Observed"].sum()
        tau = contingency_df.loc["Forecasted"].sum() / (
            contingency_df.loc["Forecasted"].sum() + contingency_df.loc["Not Forecasted"].sum()
        )
        R_score = (
            contingency_df.loc["Forecasted", "Observed"] / contingency_df["Observed"].sum()
        ) - (
            contingency_df.loc["Forecasted", "Not Observed"]
            / contingency_df["Not Observed"].sum()
        )

        threshold_row = {
            "Threshold": threshold,
            "Successful_bins": contingency_df.loc["Forecasted", "Observed"],
            "Obs_active_bins": contingency_df["Observed"].sum(),
            "tau": tau,
            "nu": nu,
            "R_score": R_score,
        }
        threshold_row_df = pandas.DataFrame([threshold_row])

        Table_molchan = pandas.concat([Table_molchan, threshold_row_df], ignore_index=True)

    bottom_row = {
        "Threshold": "Full alarms",
        "tau": 1,
        "nu": 0,
        "Obs_active_bins": contingency_df["Observed"].sum(),
    }
    top_row = {
        "Threshold": "No alarms",
        "tau": 0,
        "nu": 1,
        "Obs_active_bins": contingency_df["Observed"].sum(),
    }

    Table_molchan = pandas.concat(
        [pandas.DataFrame([top_row]), Table_molchan], ignore_index=True
    )
    Table_molchan = pandas.concat(
        [Table_molchan, pandas.DataFrame([bottom_row])], ignore_index=True
    )

    # Computation of Area Skill score (ASS)
    Tab_as_score = pandas.DataFrame()
    Tab_as_score["Threshold"] = Table_molchan["Threshold"]
    Tab_as_score["tau"] = Table_molchan["tau"]
    Tab_as_score["nu"] = Table_molchan["nu"]

    ONE = numpy.ones(len(Tab_as_score))
    Tab_as_score["CUM_BAND"] = cumulative_trapezoid(
        ONE, Tab_as_score["tau"], initial=0
    ) - cumulative_trapezoid(Tab_as_score["nu"], Tab_as_score["tau"], initial=0)
    Tab_as_score["AS_score"] = numpy.divide(
        Tab_as_score["CUM_BAND"],
        cumulative_trapezoid(ONE, Tab_as_score["tau"], initial=0) + 1e-10,
    )
    Tab_as_score.loc[Tab_as_score.index[-1], "AS_score"] = max(
        0.5, Tab_as_score["AS_score"].iloc[-1]
    )
    ASscore = numpy.round(Tab_as_score.loc[Tab_as_score.index[-1], "AS_score"], 2)

    bin = 0.01
    devstd = numpy.sqrt(1 / (12 * Table_molchan["Obs_active_bins"].iloc[0]))
    devstd = devstd * bin**-1
    devstd = numpy.ceil(devstd + 0.5)
    devstd = devstd / bin**-1
    dev_std = numpy.round(devstd, 2)

    # Plot the Molchan trajectory
    ax.plot(
        Table_molchan["tau"],
        Table_molchan["nu"],
        label=f"{forecast_label}, ASS={ASscore}±{dev_std} ",
        color=color,
        linestyle=forecast_linestyle,
    )

    # Plot uniform forecast
    if plot_uniform:
        x_uniform = numpy.arange(0, 1.001, 0.001)
        y_uniform = numpy.arange(1.00, -0.001, -0.001)
        ax.plot(x_uniform, y_uniform, linestyle="--", color="gray", label="SUP")

    # Plotting arguments
    ax.set_ylabel("Miss Rate", fontsize=label_fontsize)
    ax.set_xlabel("Fraction of area occupied by alarms", fontsize=label_fontsize)

    if linear:
        legend_loc = "upper right"
    else:
        ax.set_xscale("log")

    ax.tick_params(axis="x", labelsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)
    ax.legend(loc=legend_loc, shadow=True, fontsize=legend_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    if show:
        pyplot.show()

    return ax


###############
# Spatial plots
###############
def plot_basemap(
    basemap: Optional[str] = None,
    extent: Optional[List[float]] = None,
    coastline: bool = True,
    borders: bool = False,
    tile_depth: Union[str, int] = "auto",
    set_global: bool = False,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    ax: Optional[pyplot.Axes] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Wrapper function for multiple Cartopy base plots, including access to standard raster
    webservices.

    Args:
        basemap (str): Possible values are: 'stock_img', 'google-satellite', 'ESRI_terrain',
                       'ESRI_imagery', 'ESRI_relief', 'ESRI_topo', a custom webservice link,
                       or a GeoTiff filepath. Default is None.
        extent (list): [lon_min, lon_max, lat_min, lat_max]
        ax (matplotlib.Axes): Previously defined ax object.
        coastline (bool): Flag to plot coastline. Default True.
        borders (bool): Flag to plot country borders. Default False.
        tile_depth (str/int): Zoom level (1-12) of the basemap tiles. If 'auto', it is
            automatically derived from extent.
        set_global (bool): Display the complete globe as basemap.
        projection (cartopy.crs.Projection): Projection to be used in the basemap.
        show (bool): If True, displays the plot.

    Returns:
        matplotlib.Axes: Matplotlib Axes object.
    """

    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)

    line_autoscaler = cartopy.feature.AdaptiveScaler("110m", (("50m", 50), ("10m", 5)))
    tile_autoscaler = cartopy.feature.AdaptiveScaler(5, ((6, 50), (7, 15)))
    tile_depth = (
        4
        if set_global
        else (tile_autoscaler.scale_from_extent(extent) if tile_depth == "auto" else tile_depth)
    )
    # Add coastlines and borders
    if coastline:
        ax.coastlines(
            color=plot_args["coastline_color"], linewidth=plot_args["coastline_linewidth"]
        )
    if borders:
        borders = cartopy.feature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            line_autoscaler,
            edgecolor=plot_args["borders_color"],
            facecolor="never",
        )
        ax.add_feature(borders, linewidth=plot_args["borders_linewidth"])

    # Add basemap tiles
    try:
        if basemap == "stock_img":
            ax.stock_img()
        elif basemap is not None:
            basemap_obj = _get_basemap(basemap)
            # basemap_obj is a cartopy TILE IMAGE
            if isinstance(basemap_obj, GoogleWTS):
                ax.add_image(basemap_obj, tile_depth)
            # basemap_obj is a rasterio image
            elif isinstance(basemap_obj, DatasetReader):
                ax = rioplot.show(basemap_obj, ax=ax)

    except Exception as e:
        print(
            f"Unable to plot basemap. This might be due to no internet access. "
            f"Error: {str(e)}"
        )

    # Set up Grid-lines
    if plot_args["grid"]:
        _add_gridlines(ax, plot_args["grid_labels"], plot_args["grid_fontsize"])

    if show:
        pyplot.show()

    return ax


def plot_catalog(
    catalog: "CSEPCatalog",
    basemap: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    projection: Optional[Union[ccrs.Projection, str]] = ccrs.PlateCarree(),
    show: bool = False,
    extent: Optional[List[float]] = None,
    set_global: bool = False,
    mag_scale: float = 1,
    mag_ticks: Optional[List[float]] = None,
    plot_region: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot catalog in a region.

    Args:
        catalog (:class:`CSEPCatalog`): Catalog object to be plotted.
        basemap (str): Optional. Passed to :func:`plot_basemap` along with `kwargs`
        ax (:class:`matplotlib.pyplot.ax`): Previously defined ax object.
        show (bool): Flag if the figure is displayed.
        extent (list): Default 1.05 * :func:`catalog.region.get_bbox()`.
        projection (cartopy.crs.Projection): Projection to be used in the underlying basemap
        set_global (bool): Display the complete globe as basemap.
        mag_scale (float): Scaling of the scatter.
        mag_ticks (list): Ticks to display in the legend.
        plot_region (bool): Flag to plot the catalog region border.
        kwargs: size, alpha, markercolor, markeredgecolor, figsize, legend,
         legend_title, legend_labelspacing, legend_borderpad, legend_framealpha

    Returns:
        :class:`matplotlib.pyplot.ax` object
    """

    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    # Get spatial information for plotting
    extent = extent or _calculate_spatial_extent(catalog, set_global, plot_region)
    # Instantiate GeoAxes object
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)
    # chain basemap
    ax = plot_basemap(basemap, extent, ax=ax, set_global=set_global, show=False, **plot_args)

    # Plot catalog
    scatter = ax.scatter(
        catalog.get_longitudes(),
        catalog.get_latitudes(),
        s=_size_map(plot_args["size"], catalog.get_magnitudes(), mag_scale),
        transform=ccrs.PlateCarree(),
        color=plot_args["markercolor"],
        edgecolors=plot_args["markeredgecolor"],
        alpha=plot_args["alpha"],
    )

    # Legend
    if plot_args["legend"]:
        mw_range = [min(catalog.get_magnitudes()), max(catalog.get_magnitudes())]

        if isinstance(mag_ticks, (tuple, list, numpy.ndarray)):
            if not numpy.all([mw_range[0] <= i <= mw_range[1] for i in mag_ticks]):
                print("Magnitude ticks do not lie within the catalog magnitude range")
        elif mag_ticks is None:
            mag_ticks = numpy.linspace(mw_range[0], mw_range[1], 4)

        handles, labels = scatter.legend_elements(
            prop="sizes",
            num=list(_size_map(plot_args["size"], mag_ticks, mag_scale)),
            alpha=0.3,
        )
        ax.legend(
            handles,
            numpy.round(mag_ticks, 1),
            loc=plot_args["legend_loc"],
            handletextpad=5,
            title=plot_args.get("legend_title") or "Magnitudes",
            fontsize=plot_args["legend_fontsize"],
            title_fontsize=plot_args["legend_titlesize"],
            labelspacing=plot_args["legend_labelspacing"],
            borderpad=plot_args["legend_borderpad"],
            framealpha=plot_args["legend_framealpha"],
        )

    # Draw catalog's region border
    if plot_region:
        try:
            pts = catalog.region.tight_bbox()
            ax.plot(pts[:, 0], pts[:, 1], lw=1, color=plot_args["region_color"])
        except AttributeError:
            pass

    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"], y=1.06)

    if plot_args["tight_layout"]:
        ax.figure.tight_layout()

    if show:
        pyplot.show()

    return ax


def plot_spatial_dataset(
    gridded: numpy.ndarray,
    region: "CartesianGrid2D",
    basemap: Optional[str] = None,
    ax: Optional[pyplot.Axes] = None,
    projection: Optional[Union[ccrs.Projection, str]] = ccrs.PlateCarree(),
    show: bool = False,
    extent: Optional[List[float]] = None,
    set_global: bool = False,
    plot_region: bool = True,
    colorbar: bool = True,
    colormap: Union[str, matplotlib.colors.Colormap] = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    clabel: Optional[str] = None,
    alpha: Optional[float] = None,
    alpha_exp: float = 0,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot spatial dataset such as data from a gridded forecast.

    Args:
        gridded (numpy.ndarray): 2D array of values corresponding to the region.
        region (CartesianGrid2D): Region in which gridded values are contained.
        basemap (str): Optional. Passed to :func:`plot_basemap` along with `kwargs`
        ax (Optional[pyplot.Axes]): Previously defined ax object.
        projection (cartopy.crs.Projection): Projection to be used in the basemap.
        show (bool): If True, displays the plot.
        extent (Optional[List[float]]): [lon_min, lon_max, lat_min, lat_max].
        set_global (bool): Display the complete globe as basemap.
        plot_region (bool): If True, plot the dataset region border.
        colorbar (bool): If True, include a colorbar.
        colormap (Union[str, matplotlib.colors.Colormap]): Colormap to use.
        clim (Optional[Tuple[float, float]]): Range of the colorbar.
        clabel (Optional[str]): Label of the colorbar.
        alpha (Optional[float]): Transparency level.
        alpha_exp (float): Exponent for the alpha function (recommended between 0.4 and 1).
        kwargs: colorbar_labelsize, colorbar_ticksize
    Returns:
        matplotlib.axes.Axes: Matplotlib axes handle.
    """

    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}

    # Get spatial information for plotting
    extent = extent or _calculate_spatial_extent(region, set_global, plot_region)
    # Instantiate GeoAxes object
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)

    # chain basemap
    ax = plot_basemap(basemap, extent, ax=ax, set_global=set_global, show=False, **plot_args)

    # Define colormap and alpha transparency
    colormap, alpha = _define_colormap_and_alpha(colormap, alpha_exp, alpha)

    # Plot spatial dataset
    lons, lats = numpy.meshgrid(
        numpy.append(region.xs, region.get_bbox()[1]),
        numpy.append(region.ys, region.get_bbox()[3]),
    )
    im = ax.pcolor(
        lons, lats, gridded, cmap=colormap, alpha=alpha, snap=True, transform=ccrs.PlateCarree()
    )
    im.set_clim(clim)

    # Colorbar options
    if colorbar:
        _add_colorbar(
            ax, im, clabel, plot_args["colorbar_labelsize"], plot_args["colorbar_ticksize"]
        )

    # Draw forecast's region border
    if plot_region and not set_global:
        try:
            pts = region.tight_bbox()
            ax.plot(pts[:, 0], pts[:, 1], lw=1, color=plot_args["region_color"])
        except AttributeError:
            pass

    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"], y=1.06)

    if show:
        pyplot.show()

    return ax


#####################
# Plot helper functions
#####################
def _get_marker_style(obs_stat, p, one_sided_lower):
    """Returns matplotlib marker style as fmt string"""
    if obs_stat < p[0] or obs_stat > p[1]:
        # red circle
        fmt = "ro"
    else:
        # green square
        fmt = "gs"
    if one_sided_lower:
        if obs_stat < p[0]:
            fmt = "ro"
        else:
            fmt = "gs"
    return fmt


def _get_marker_t_color(distribution):
    """Returns matplotlib marker style as fmt string"""
    if distribution[0] > 0.0 and distribution[1] > 0.0:
        fmt = "green"
    elif distribution[0] < 0.0 and distribution[1] < 0.0:
        fmt = "red"
    else:
        fmt = "grey"

    return fmt


def _get_marker_w_color(distribution, percentile):
    """Returns matplotlib marker style as fmt string"""

    if distribution < (1 - percentile / 100):
        fmt = True
    else:
        fmt = False

    return fmt


def _get_axis_limits(pnts, border=0.05):
    """Returns a tuple of x_min and x_max given points on plot."""
    x_min = numpy.min(pnts)
    x_max = numpy.max(pnts)
    xd = (x_max - x_min) * border
    return x_min - xd, x_max + xd


def _get_basemap(basemap):
    last_cache = os.path.join(os.path.dirname(cartopy.config["cache_dir"]), 'last_cartopy_cache')

    def _clean_cache(basemap_):
        if os.path.isfile(last_cache):
            with open(last_cache, 'r') as fp:
                cache_src = fp.read()
            if cache_src != basemap_:
                if os.path.isdir(cartopy.config["cache_dir"]):
                    print(f'Cleaning existing {basemap_} cache')
                    shutil.rmtree(cartopy.config["cache_dir"])

    def _save_cache_src(basemap_):
        with open(last_cache, 'w') as fp:
            fp.write(basemap_)

    cache = True

    warning_message_to_suppress = ('Cartopy created the following directory to cache'
                                   ' GoogleWTS tiles')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=warning_message_to_suppress)
        if basemap == "google-satellite":
            _clean_cache(basemap)
            tiles = img_tiles.GoogleTiles(style="satellite", cache=cache)
            _save_cache_src(basemap)

        elif basemap == "ESRI_terrain":
            _clean_cache(basemap)
            webservice = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/"
                "MapServer/tile/{z}/{y}/{x}.jpg"
            )
            tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
            _save_cache_src(basemap)

        elif basemap == "ESRI_imagery":
            _clean_cache(basemap)
            webservice = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                "MapServer/tile/{z}/{y}/{x}.jpg"
            )
            tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
            _save_cache_src(basemap)

        elif basemap == "ESRI_relief":
            _clean_cache(basemap)
            webservice = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/"
                "MapServer/tile/{z}/{y}/{x}.jpg"
            )
            tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
            _save_cache_src(basemap)

        elif basemap == "ESRI_topo":
            _clean_cache(basemap)
            webservice = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/"
                "MapServer/tile/{z}/{y}/{x}.jpg"
            )
            tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
            _save_cache_src(basemap)

        elif os.path.isfile(basemap):
            return rasterio.open(basemap)

        else:
            try:
                _clean_cache(basemap)
                webservice = basemap
                tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
                _save_cache_src(basemap)
            except Exception as e:
                raise ValueError(f"Basemap type not valid or not implemented. {e}")

    return tiles


def _add_labels_for_publication(figure, style="bssa", labelsize=16):
    """Adds publication labels too the outside of a figure."""
    all_axes = figure.get_axes()
    ascii_iter = iter(string.ascii_lowercase)
    for ax in all_axes:
        # check for colorbar and ignore for annotations
        if ax.get_label() == "Colorbar":
            continue
        annot = next(ascii_iter)
        if style == "bssa":
            ax.annotate(
                f"({annot})", (0.025, 1.025), xycoords="axes fraction", fontsize=labelsize
            )

    return


def _plot_pvalues_and_intervals(test_results, ax, var=None):
    """Plots p-values and intervals for a list of Poisson or NBD test results

    Args:
        test_results (list): list of EvaluationResults for N-test. All tests should use the same
                             distribution (ie Poisson or NBD).
        ax (matplotlib.axes.Axes.axis): axes to use for plot. create using matplotlib
        var (float): variance of the NBD distribution. Must be used for NBD plots.

    Returns:
        ax (matplotlib.axes.Axes.axis): axes handle containing this plot

    Raises:
        ValueError: throws error if NBD tests are supplied without a variance
    """

    variance = var
    percentile = 97.5
    p_values = []

    # Differentiate between N-tests and other consistency tests
    if test_results[0].name == "NBD N-Test" or test_results[0].name == "Poisson N-Test":
        legend_elements = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="red",
                lw=0,
                label=r"p < 10e-5",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="#FF7F50",
                lw=0,
                label=r"10e-5 $\leq$ p < 10e-4",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="gold",
                lw=0,
                label=r"10e-4 $\leq$ p < 10e-3",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                lw=0,
                label=r"10e-3 $\leq$ p < 0.0125",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="skyblue",
                lw=0,
                label=r"0.0125 $\leq$ p < 0.025",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="blue",
                lw=0,
                label=r"p $\geq$ 0.025",
                markersize=10,
                markeredgecolor="k",
            ),
        ]
        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor="k")
        # Act on Negative binomial tests
        if test_results[0].name == "NBD N-Test":
            if var is None:
                raise ValueError("var must not be None if N-tests use the NBD distribution.")

            for i in range(len(test_results)):
                mean = test_results[i].test_distribution[1]
                upsilon = 1.0 - ((variance - mean) / variance)
                tau = mean**2 / (variance - mean)
                phigh97 = scipy.stats.nbinom.ppf((1 - percentile / 100.0) / 2.0, tau, upsilon)
                plow97 = scipy.stats.nbinom.ppf(
                    1 - (1 - percentile / 100.0) / 2.0, tau, upsilon
                )
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    xerr=numpy.array([[low97, high97]]).T,
                    capsize=4,
                    color="slategray",
                    alpha=1.0,
                    zorder=0,
                )
                p_values.append(
                    test_results[i].quantile[1] * 2.0
                )  # Calculated p-values according to Meletti et al., (2021)

                if p_values[i] < 10e-5:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="red",
                        markersize=8,
                        zorder=2,
                    )
                if p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="#FF7F50",
                        markersize=8,
                        zorder=2,
                    )
                if p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="gold",
                        markersize=8,
                        zorder=2,
                    )
                if p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="white",
                        markersize=8,
                        zorder=2,
                    )
                if p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="skyblue",
                        markersize=8,
                        zorder=2,
                    )
                if p_values[i] >= 0.025:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="blue",
                        markersize=8,
                        zorder=2,
                    )
        # Act on Poisson N-test
        if test_results[0].name == "Poisson N-Test":
            for i in range(len(test_results)):
                plow97 = scipy.stats.poisson.ppf(
                    (1 - percentile / 100.0) / 2.0, test_results[i].test_distribution[1]
                )
                phigh97 = scipy.stats.poisson.ppf(
                    1 - (1 - percentile / 100.0) / 2.0, test_results[i].test_distribution[1]
                )
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    xerr=numpy.array([[low97, high97]]).T,
                    capsize=4,
                    color="slategray",
                    alpha=1.0,
                    zorder=0,
                )
                p_values.append(test_results[i].quantile[1] * 2.0)
                if p_values[i] < 10e-5:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="red",
                        markersize=8,
                        zorder=2,
                    )
                elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="#FF7F50",
                        markersize=8,
                        zorder=2,
                    )
                elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="gold",
                        markersize=8,
                        zorder=2,
                    )
                elif p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="white",
                        markersize=8,
                        zorder=2,
                    )
                elif p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="skyblue",
                        markersize=8,
                        zorder=2,
                    )
                elif p_values[i] >= 0.025:
                    ax.plot(
                        test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        marker="o",
                        color="blue",
                        markersize=8,
                        zorder=2,
                    )
    # Operate on all other consistency tests
    else:
        for i in range(len(test_results)):
            plow97 = numpy.percentile(test_results[i].test_distribution, 2.5)
            phigh97 = numpy.percentile(test_results[i].test_distribution, 97.5)
            low97 = test_results[i].observed_statistic - plow97
            high97 = phigh97 - test_results[i].observed_statistic
            ax.errorbar(
                test_results[i].observed_statistic,
                (len(test_results) - 1) - i,
                xerr=numpy.array([[low97, high97]]).T,
                capsize=4,
                color="slategray",
                alpha=1.0,
                zorder=0,
            )
            p_values.append(test_results[i].quantile)

            if p_values[i] < 10e-5:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="red",
                    markersize=8,
                    zorder=2,
                )
            elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="#FF7F50",
                    markersize=8,
                    zorder=2,
                )
            elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="gold",
                    markersize=8,
                    zorder=2,
                )
            elif p_values[i] >= 10e-3 and p_values[i] < 0.025:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="white",
                    markersize=8,
                    zorder=2,
                )
            elif p_values[i] >= 0.025 and p_values[i] < 0.05:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="skyblue",
                    markersize=8,
                    zorder=2,
                )
            elif p_values[i] >= 0.05:
                ax.plot(
                    test_results[i].observed_statistic,
                    (len(test_results) - 1) - i,
                    marker="o",
                    color="blue",
                    markersize=8,
                    zorder=2,
                )

        legend_elements = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="red",
                lw=0,
                label=r"p < 10e-5",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="#FF7F50",
                lw=0,
                label=r"10e-5 $\leq$ p < 10e-4",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="gold",
                lw=0,
                label=r"10e-4 $\leq$ p < 10e-3",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                lw=0,
                label=r"10e-3 $\leq$ p < 0.025",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="skyblue",
                lw=0,
                label=r"0.025 $\leq$ p < 0.05",
                markersize=10,
                markeredgecolor="k",
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="blue",
                lw=0,
                label=r"p $\geq$ 0.05",
                markersize=10,
                markeredgecolor="k",
            ),
        ]

        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor="k")

    return ax


def _autosize_scatter(markersize, values, scale):
    if isinstance(scale, (int, float)):
        # return (values - min(values) + markersize) ** scale  # Adjust this formula as needed for better visualization
        # return mark0ersize * (1 + (values - numpy.min(values)) / (numpy.max(values) - numpy.min(values)) ** scale)
        return markersize / (scale ** min(values)) * numpy.power(values, scale)

    elif isinstance(scale, (numpy.ndarray, list)):
        return scale
    else:
        raise ValueError("scale data type not supported")


def _size_map(markersize, values, scale):
    if isinstance(scale, (int, float)):
        return markersize / (scale ** min(values)) * numpy.power(values, scale)
    elif isinstance(scale, (numpy.ndarray, list)):
        return scale
    else:
        raise ValueError("Scale data type not supported")


def _autoscale_histogram(ax: pyplot.Axes, bin_edges, simulated, observation, mass=99.5):

    upper_xlim = numpy.percentile(simulated, 100 - (100 - mass) / 2)
    upper_xlim = numpy.max([upper_xlim, numpy.max(observation)])
    d_bin = bin_edges[1] - bin_edges[0]
    upper_xlim = upper_xlim + 2 * d_bin

    lower_xlim = numpy.percentile(simulated, (100 - mass) / 2)
    lower_xlim = numpy.min([lower_xlim, numpy.min(observation)])
    lower_xlim = lower_xlim - 2 * d_bin

    try:
        ax.set_xlim([lower_xlim, upper_xlim])
    except ValueError:
        print("Ignoring observation in axis scaling because inf or -inf")
        upper_xlim = numpy.percentile(simulated, 99.75)
        upper_xlim = upper_xlim + 2 * d_bin

        lower_xlim = numpy.percentile(simulated, 0.25)
        lower_xlim = lower_xlim - 2 * d_bin

        ax.set_xlim([lower_xlim, upper_xlim])

    return ax


def _annotate_distribution_plot(
    ax, evaluation_result, auto_annotate, plot_args
) -> matplotlib.axes.Axes:
    """Returns specific plot details based on the type of evaluation_result."""

    annotation_text = None
    annotation_xy = None
    title = None
    xlabel = None
    ylabel = None

    if auto_annotate:
        if evaluation_result.name == "Catalog N-Test":
            xlabel = "Event Count"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.5, 0.3)
            if isinstance(evaluation_result.quantile, (list, np.ndarray)):
                annotation_text = (
                    f"$\\delta_1 = P(X \\geq x) = {evaluation_result.quantile[0]:.2f}$\n"
                    f"$\\delta_2 = P(X \\leq x) = {evaluation_result.quantile[1]:.2f}$\n"
                    f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
                )
            else:
                annotation_text = (
                    f"$\\gamma = P(X \\leq x) = {evaluation_result.quantile:.2f}$\n"
                    f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
                )

        elif evaluation_result.name == "Catalog S-Test":
            xlabel = "Normalized Spatial Statistic"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.2, 0.6)
            annotation_text = (
                f"$\\gamma = P(X \\leq x) = {numpy.array(evaluation_result.quantile).ravel()[-1]:.2f}$\n"
                f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
            )

        elif evaluation_result.name == "Catalog M-Test":
            xlabel = "Magnitude"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.55, 0.6)
            annotation_text = (
                f"$\\gamma = P(X \\geq x) = {numpy.array(evaluation_result.quantile).ravel()[0]:.2f}$\n"
                f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
            )
        elif evaluation_result.name == "Catalog PL-Test":
            xlabel = "Likelihood"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.55, 0.3)
            annotation_text = (
                f"$\\gamma = P(X \\leq x) = {numpy.array(evaluation_result.quantile).ravel()[-1]:.2f}$\n"
                f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
            )

        else:
            xlabel = "Statistic"
            ylabel = "Number of Catalogs"

    if annotation_text or plot_args.get("annotation_text"):
        ax.annotate(
            annotation_text or plot_args.get("annotation_text"),
            annotation_xy or plot_args.get("annotation_xy"),
            xycoords="axes fraction",
            fontsize=plot_args.get("annotation_fontsize"),
        )
    ax.set_xlabel(plot_args.get("xlabel") or xlabel)
    ax.set_ylabel(plot_args.get("ylabel") or ylabel)
    ax.set_title(plot_args["title"] or title)

    return ax


def _calculate_spatial_extent(catalog, set_global, region_border, padding_fraction=0.05):
    # todo: perhaps calculate extent also from chained ax object
    bbox = catalog.get_bbox()
    if region_border:
        try:
            bbox = catalog.region.get_bbox()
        except AttributeError:
            pass

    if set_global:
        return None

    dh = (bbox[1] - bbox[0]) * padding_fraction
    dv = (bbox[3] - bbox[2]) * padding_fraction
    return [bbox[0] - dh, bbox[1] + dh, bbox[2] - dv, bbox[3] + dv]


def _create_geo_axes(figsize, extent, projection, set_global):

    if projection == "approx":
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        central_latitude = (extent[2] + extent[3]) / 2.0
        # Set plot aspect according to local longitude-latitude ratio in metric units
        LATKM = 110.574  # length of a ° of latitude [km] --> ignores Earth's flattening
        ax.set_aspect(LATKM / (111.320 * numpy.cos(numpy.deg2rad(central_latitude))))
    else:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    if set_global:
        ax.set_global()
    elif extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def _calculate_marker_size(markersize, magnitudes, scale):
    mw_range = [min(magnitudes), max(magnitudes)]
    if isinstance(scale, (int, float)):
        return (markersize / (scale ** mw_range[0])) * numpy.power(magnitudes, scale)
    elif isinstance(scale, (numpy.ndarray, list)):
        return scale
    else:
        raise ValueError("Scale data type not supported")


# Helper function to add gridlines
def _add_gridlines(ax, grid_labels, grid_fontsize):
    gl = ax.gridlines(draw_labels=grid_labels, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style["fontsize"] = grid_fontsize
    gl.ylabel_style["fontsize"] = grid_fontsize
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def _define_colormap_and_alpha(cmap, alpha_exp, alpha_0=None):
    """
    Defines the colormap and applies alpha transparency based on the given parameters.

    Args:
        cmap (str or matplotlib.colors.Colormap): The colormap to be used.
        alpha_0 (float or None): If set, this alpha will be applied uniformly across the colormap.
        alpha_exp (float): Exponent to control transparency scaling. If set to 0, no alpha scaling is applied.

    Returns:
        cmap (matplotlib.colors.ListedColormap): The resulting colormap with applied alpha.
        alpha (float or None): The alpha value used for the entire colormap, or None if alpha is scaled per color.
    """

    # Get the colormap object if a string is provided
    if isinstance(cmap, str):
        cmap = matplotlib.pyplot.get_cmap(cmap)

    cmap_tup = cmap(numpy.arange(cmap.N))

    if alpha_0 is not None:
        cmap_tup[:, -1] = alpha_0
        alpha = alpha_0
    else:
        if alpha_exp != 0:
            cmap_tup[:, -1] = numpy.linspace(0, 1, cmap.N) ** alpha_exp
            alpha = None
        else:
            alpha = 1

    cmap = matplotlib.colors.ListedColormap(cmap_tup)

    return cmap, alpha


def _add_colorbar(ax, im, clabel, clabel_fontsize, cticks_fontsize):
    fig = ax.get_figure()
    cax = fig.add_axes(
        [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.025, ax.get_position().height],
        label="Colorbar",
    )
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.set_label(clabel, fontsize=clabel_fontsize)
    cbar.ax.tick_params(labelsize=cticks_fontsize)


def _process_stat_distribution(res, percentile, variance, normalize, one_sided_lower):
    """Process the distribution based on its type and return plotting values."""
    dist_type = res.test_distribution[0]

    if dist_type == "poisson":
        mean = res.test_distribution[1]
        plow = scipy.stats.poisson.ppf((1 - percentile / 100.0) / 2.0, mean)
        phigh = scipy.stats.poisson.ppf(1 - (1 - percentile / 100.0) / 2.0, mean)
        observed_statistic = res.observed_statistic
    elif dist_type == "negative_binomial":
        mean = res.test_distribution[1]
        upsilon = 1.0 - ((variance - mean) / variance)
        tau = mean**2 / (variance - mean)
        plow = scipy.stats.nbinom.ppf((1 - percentile / 100.0) / 2.0, tau, upsilon)
        phigh = scipy.stats.nbinom.ppf(1 - (1 - percentile / 100.0) / 2.0, tau, upsilon)
        observed_statistic = res.observed_statistic

    else:
        if normalize:
            test_distribution = numpy.array(res.test_distribution) - res.observed_statistic
            observed_statistic = 0
        else:
            test_distribution = numpy.array(res.test_distribution)
            observed_statistic = res.observed_statistic

        if one_sided_lower:
            plow = numpy.percentile(test_distribution, 100 - percentile)
            phigh = numpy.percentile(test_distribution, 100)
        else:
            plow = numpy.percentile(test_distribution, (100 - percentile) / 2.0)
            phigh = numpy.percentile(test_distribution, 100 - (100 - percentile) / 2.0)
        mean = numpy.mean(test_distribution)

    return plow, phigh, mean, observed_statistic
