import os
import shutil
import warnings
from typing import TYPE_CHECKING, Optional, Any, List, Union, Tuple, Sequence, Dict

# Third-party imports
import numpy
import pandas
import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as pyplot
from cartopy.io import img_tiles
from cartopy.io.img_tiles import GoogleWTS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.lines import Line2D
from rasterio import DatasetReader
from rasterio import plot as rio_plot
from rasterio import open as rio_open
from scipy.integrate import cumulative_trapezoid
from scipy.stats import poisson, nbinom, beta

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
    "secondary_color": "red",
    "alpha": 0.8,
    "linewidth": 1,
    "linestyle": "-",
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
    # Spatial plotting
    "grid_labels": True,
    "grid_fontsize": 8,
    "region_color": "black",
    "coastline": True,
    "coastline_color": "black",
    "coastline_linewidth": 1.5,
    "borders_color": "black",
    "borders_linewidth": 1.5,
    # Color bars
    "colorbar_labelsize": 12,
    "colorbar_ticksize": 10,
}


########################
# Data-exploratory plots
########################
def plot_magnitude_versus_time(
    catalog: "CSEPCatalog",
    color: str = "steelblue",
    size: int = 4,
    max_size: int = 300,
    power: int = 4,
    alpha: float = 0.5,
    reset_times: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Scatter plot of the catalog magnitudes and origin times. The size of each event is scaled
    exponentially by its magnitude using the parameters ``size``,``max_size`` and ``power``.

    Args:
        catalog (CSEPCatalog): Catalog of seismic events to be plotted.
        color (str, optional): Color of the scatter plot. Defaults to `'steelblue'`.
        size (int, optional): Marker size for the event with the minimum magnitude. Defaults
            to `4`.
        max_size (int, optional): Marker size for the event with the maximum magnitude.
            Defaults to `300`.
        power (int, optional): Power scaling of the scatter sizing. Defaults to `4`.
        alpha (float, optional): Transparency level for the scatter points. Defaults to `0.5`.
        reset_times (bool): If True, x-axis shows time in days since first event.
        ax (matplotlib.axes.Axes, optional): Axis object on which to plot. If not provided, a
            new figure and axis are created. Defaults to `None`.
        show (bool, optional): Whether to display the plot. Defaults to `False`.
        **kwargs:
            Additional keyword arguments to customize the plot:

            - **figsize** (`tuple`): The size of the figure.
            - **title** (`str`): Plot title. Defaults to `None`.
            - **title_fontsize** (`int`): Font size for the plot title.
            - **xlabel** (`str`): Label for the X-axis. Defaults to `'Datetime'`.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis. Defaults to `'Magnitude'`.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **datetime_locator** (`matplotlib.dates.Locator`): Locator for the
              X-axis datetime ticks.
            - **datetime_formatter** (`str` or `matplotlib.dates.Formatter`):
              Formatter for the datetime axis. Defaults to `'%Y-%m-%d'`.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **tight_layout** (`bool`): Whether to use a tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object with the plotted data.
    """
    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_magnitude_versus_time.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Plot data
    mag = catalog.data["magnitude"]
    datetimes = catalog.get_datetimes()

    if reset_times:
        # Convert to days since first event
        SECONDS_PER_DAY = 86400
        timestamps = numpy.array([dt.timestamp() for dt in datetimes])
        xdata = (timestamps - timestamps[0]) / SECONDS_PER_DAY
        xlabel = plot_args["xlabel"] or "Days since first event"
    else:
        xdata = datetimes
        xlabel = plot_args["xlabel"] or "Datetime"

    ax.scatter(
        xdata,
        mag,
        marker="o",
        c=color,
        s=_autosize_scatter(mag, min_size=size, max_size=max_size, power=power),
        alpha=alpha,
    )

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=plot_args["xlabel_fontsize"])
    ax.set_ylabel(plot_args["ylabel"] or "Magnitude", fontsize=plot_args["ylabel_fontsize"])
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])

    # Autoformat ticks and labels
    if not reset_times:
        ax.xaxis.set_major_locator(plot_args["datetime_locator"])
        ax.xaxis.set_major_formatter(plot_args["datetime_formatter"])
        fig.autofmt_xdate()
    ax.grid(plot_args["grid"])

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


def _plot_cumulative_events_versus_time(
    catalog_forecast: "CatalogForecast",
    observation: "CSEPCatalog",
    time_axis: str = "datetime",
    bins: int = 50,
    sim_label: Optional[str] = "Simulated",
    obs_label: Optional[str] = "Observation",
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots the cumulative number of forecasted events from a
    :class:`~csep.core.forecasts.CatalogForecast` versus the observed events over time.

    Args:
        catalog_forecast (CatalogForecast): The forecasted synthetic catalogs.
        observation (CSEPCatalog): The observed catalog.
        time_axis (str, optional): The type of time axis (`'datetime'`, `'days'`, `'hours'`).
            Defaults to `'datetime'`.
        bins (int, optional): The number of bins for time slicing. Defaults to `50`.
        sim_label (str, optional): Label for simulated data. Defaults to `'Simulated'`.
        obs_label (str, optional): Label for observed data. Defaults to `'Observation'`.
        ax (matplotlib.axes.Axes, optional): Axis object on which to plot. If not provided, a
            new figure and axis are created. Defaults to `None`.
        show (bool, optional): If True, displays the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments to customize the plot:

            - **figsize** (`tuple`): The size of the figure.
            - **xlabel** (`str`): Label for the X-axis. Defaults to `'Datetime'`,
              `'Days after Mainshock'`, or `'Hours after Mainshock'`, depending on
              `time_axis`.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis. Defaults to
              `'Cumulative event counts'`.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **title** (`str`): Title of the plot. Defaults to `None`.
            - **title_fontsize** (`int`): Font size for the plot title.
            - **color** (`str`): Color for the simulated forecast.
            - **linewidth** (`float`): Line width for the plot lines. Defaults to
              `1.5`.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **legend** (`bool`): Whether to show the legend. Defaults to `True`.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.
            - **datetime_locator** (`matplotlib.dates.Locator`): Locator for the
              X-axis datetime ticks.
            - **datetime_formatter** (`str` or `matplotlib.dates.Formatter`):
              Formatter for the datetime axis. Defaults to ``'%Y-%m-%d'``.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object with the plotted data.
    """
    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_cumulative_events_versus_time.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
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
        catalog_forecast (CatalogForecast or list of CSEPCatalog): A catalog-based forecast
            or a list of observed catalogs.
        observation (CSEPCatalog): The observed catalog for comparison.
        magnitude_bins (list of float or numpy.ndarray, optional): The bins for magnitude
            histograms. If `None`, defaults to the region magnitudes or standard CSEP bins.
            Defaults to `None`.
        percentile (int, optional): The percentile used for uncertainty intervals. Defaults to
            `95`.
        ax (matplotlib.axes.Axes, optional): The axes object to draw the plot on. If `None`, a
            new figure and axes are created. Defaults to `None`.
        show (bool, optional): Whether to display the plot immediately. Defaults to `False`.
        **kwargs: Additional keyword arguments to customize the plot:

            - **figsize** (`tuple`): The size of the figure.
            - **color** (`str`): Color for the observed histogram points.
            - **markersize** (`int`): Size of the markers in the plot. Defaults to `6`.
            - **capsize** (`float`): Size of the error bar caps. Defaults to `4`.
            - **linewidth** (`float`): Width of the error bar lines. Defaults to `1.5`.
            - **xlim** (`tuple`): Limits for the X-axis.
            - **xlabel** (`str`): Label for the X-axis. Defaults to `'Magnitude'`.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis. Defaults to `'Event count'`.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **title** (`str`): Title of the plot. Defaults to `'Magnitude Histogram'`.
            - **title_fontsize** (`int`): Font size for the plot title.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_magnitude_histogram.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
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


###############
# Spatial plots
###############
def _plot_basemap(
    basemap: Optional[str] = None,
    extent: Optional[List[float]] = None,
    coastline: bool = True,
    borders: bool = False,
    tile_depth: Union[str, int] = "auto",
    set_global: bool = False,
    projection: Optional[ccrs.Projection] = None,
    ax: Optional[pyplot.Axes] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Wrapper function for multiple Cartopy base plots, including access to standard raster
    web services and filesystem geoTIFF.

    Args:
        basemap (str): Possible values are: `'stock_img'`, `'google-satellite'`,
            `'ESRI_terrain'`, `'ESRI_imagery'`, `'ESRI_relief'`, `'ESRI_topo'`, a custom web
            service link, or a GeoTiff filepath. Defaults to `None`.
        extent (list of float): `[lon_min, lon_max, lat_min, lat_max]`. Defaults to `None`.
        coastline (bool): Flag to plot coastline. Defaults to `True`.
        borders (bool): Flag to plot country borders. Defaults to `False`.
        tile_depth (str or int): Zoom resolution level (`1-12`) of the basemap tiles. If
            `'auto'`, it is automatically derived from extent. Defaults to `'auto'`.
        set_global (bool): Display the complete globe as basemap. Required by global forecasts.
            Defaults to `False`.
        projection (cartopy.crs.Projection): Projection to be used in the basemap. It can be a
            cartopy projection instance, or `approx` for a quick approximation of Mercator.
            Defaults to :class:`~cartopy.crs.PlateCarree` if `None`.
        ax (matplotlib.axes.Axes, optional): Previously defined ax object. If `None`, a new
            axis is created. Defaults to `None`.
        show (bool): If `True`, displays the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments to customize the plot:

            - **figsize** (`tuple`): The size of the figure.
            - **coastline_color** (`str`): Color for the coastlines.
            - **coastline_linewidth** (`float`): Line width for the coastlines.
            - **borders_color** (`str`): Color for the borders.
            - **borders_linewidth** (`float`): Line width for the borders.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **grid_labels** (`bool`): Whether to show grid labels.
            - **grid_fontsize** (`int`): Font size for the grid labels.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object with the plotted data.
    """

    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs}
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)

    line_autoscaler = cartopy.feature.AdaptiveScaler("110m", (("50m", 50), ("10m", 5)))
    tile_autoscaler = cartopy.feature.AdaptiveScaler(5, ((6, 50), (7, 15), (8, 5), (9, 1)))
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
                ax = rio_plot.show(basemap_obj, ax=ax)

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
    projection: Optional[Union[ccrs.Projection, str]] = None,
    extent: Optional[Sequence[float]] = None,
    set_global: bool = False,
    mag_ticks: Optional[Union[Sequence[float], numpy.ndarray, int]] = None,
    size: float = 15,
    max_size: float = 300,
    power: float = 3,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    plot_region: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Spatial plot of catalog. Can be plotted over a basemap if desired, by passing the keyword
    parameters of the function :func:`~csep.utils.plots.plot_basemap`. The size of the events
    is automatically scaled according to their magnitude. Fine-tuning of an exponential sizing
    function can be set with the parameters ``size``, ``max_size``, ``power``, ``min_val`` and
    ``max_val``.

    Args:
        catalog (CSEPCatalog): Catalog object to be plotted.
        basemap (str): Passed to :func:`~csep.utils.plots.plot_basemap` along with `kwargs`.
            Possible values are: `'stock_img'`, `'google-satellite'`, `'ESRI_terrain'`,
            `'ESRI_imagery'`, `'ESRI_relief'`, `'ESRI_topo'`, a custom web service link, or a
            GeoTiff filepath. Defaults to `None`.
        projection (cartopy.crs.Projection or str): Projection to be used in the underlying
            basemap. Can be a cartopy projection instance, or `approx` for a quick approximation
             of Mercator. Defaults to :class:`~cartopy.crs.PlateCarree` if `None`.
        extent (list of float): Defaults to `1.05` * :meth:`catalog.region.get_bbox`. Defaults
            to `None`.
        set_global (bool): Display the complete globe. Defaults to `False`.
        mag_ticks (list of float, int): Ticks to display in the legend. Can be an array/list of
            magnitudes, or a number of bins to discretize the magnitude range. Defaults to
            `None`.
        size (float): Size of the minimum magnitude event. Defaults to `15`.
        max_size (float): Size of the maximum magnitude event. Defaults to `300`.
        power (float): Power scaling of the scatter sizing. Defaults to `3`.
        min_val (float): Override minimum magnitude of the catalog for scatter sizing.  Useful
            to plot multiple catalogs with different magnitude ranges. Defaults to `None`.
        max_val (float): Override maximum magnitude of the catalog for scatter sizing. Useful
            to plot multiple catalogs with different magnitude ranges. Defaults to `None`.
        plot_region (bool): Flag to plot the catalog region border. Defaults to `False`.
        ax (matplotlib.axes.Axes): Previously defined ax object. Defaults to `None`.
        show (bool): If `True`, displays the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments to customize the plot:

            - **alpha** (`float`): Transparency level for the scatter points.
            - **markercolor** (`str`): Color for the scatter points.
            - **markeredgecolor** (`str`): Color for the edges of the scatter points.
            - **figsize** (`tuple`): The size of the figure.
            - **legend** (`bool`): Whether to display a legend. Defaults to `True`.
            - **legend_title** (`str`): Title for the legend.
            - **legend_labelspacing** (`float`): Spacing between labels in the legend.
            - **legend_borderpad** (`float`): Border padding for the legend.
            - **legend_framealpha** (`float`): Frame alpha for the legend.
            - **region_color** (`str`): Color for the region border.
            - **title** (`str`): Title of the plot.
            - **title_fontsize** (`int`): Font size of the plot title.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object with the plotted data.
    """

    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_catalog.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    # Get spatial information for plotting
    extent = extent or _calculate_spatial_extent(catalog, set_global, plot_region)
    # Instantiate GeoAxes object
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)
    # chain basemap
    ax = plot_basemap(basemap, extent, ax=ax, set_global=set_global, show=False, **plot_args)

    # Plot catalog
    ax.scatter(
        catalog.get_longitudes(),
        catalog.get_latitudes(),
        s=_autosize_scatter(
            values=catalog.get_magnitudes(),
            min_size=size,
            max_size=max_size,
            power=power,
            min_val=min_val,
            max_val=max_val,
        ),
        transform=ccrs.PlateCarree(),
        color=plot_args["markercolor"],
        edgecolors=plot_args["markeredgecolor"],
        alpha=plot_args["alpha"],
    )

    # Legend
    if plot_args["legend"]:
        if isinstance(mag_ticks, (list, numpy.ndarray)):
            mag_ticks = numpy.array(mag_ticks)
        else:
            mw_range = [min(catalog.get_magnitudes()), max(catalog.get_magnitudes())]
            mag_ticks = numpy.linspace(mw_range[0], mw_range[1], mag_ticks or 4, endpoint=True)

        # Map mag_ticks to marker sizes using the custom size mapping function
        legend_sizes = _autosize_scatter(
            values=mag_ticks,
            min_size=size,
            max_size=max_size,
            power=power,
            min_val=min_val or numpy.min(catalog.get_magnitudes()),
            max_val=max_val or numpy.max(catalog.get_magnitudes()),
        )

        # Create custom legend handles
        handles = [
            pyplot.Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                label=str(m),
                markersize=numpy.sqrt(s),
                markerfacecolor="gray",
                alpha=0.5,
                markeredgewidth=0.8,
                markeredgecolor="black",
            )
            for m, s in zip(mag_ticks, legend_sizes)
        ]

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


def plot_gridded_dataset(
    gridded: numpy.ndarray,
    region: "CartesianGrid2D",
    basemap: Optional[str] = None,
    ax: Optional[pyplot.Axes] = None,
    projection: Optional[Union[ccrs.Projection, str]] = None,
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
    Plot spatial gridded dataset such as data from a gridded forecast. Can be plotted over a
    basemap if desired, by passing the keyword parameters of the function
    :func:`~csep.utils.plots.plot_basemap`. The color map can be fine-tuned using the arguments
    ``colorbar``, ``colormap``, ``clim``. An exponential transparency function can be set
    with ``alpha`` and ``alpha_exp``.

    Args:
        gridded (numpy.ndarray): 2D-square array of values corresponding to the region. See
            :class:`~csep.core.regions.CartesianGrid2D` and
            :meth:`~csep.core.regions.CartesianGrid2D.get_cartesian` to convert a 1D array
            (whose indices correspond to each spatial cell) to a 2D-square array.
        region (CartesianGrid2D): Region in which gridded values are contained.
        basemap (str): Passed to :func:`~csep.utils.plots.plot_basemap` along with `kwargs`.
        ax (matplotlib.axes.Axes): Previously defined ax object. Defaults to `None`.
        projection (cartopy.crs.Projection or str): Projection to be used in the underlying
            basemap. It can be a cartopy projection instance, or `approx` for a quick
            approximation of Mercator. Defaults to :class:`~cartopy.crs.PlateCarree` if `None`.
        show (bool): If `True`, displays the plot. Defaults to `False`.
        extent (list of float): ``[lon_min, lon_max, lat_min, lat_max]``. Defaults to `None`.
        set_global (bool): Display the complete globe as basemap. Defaults to `False`.
        plot_region (bool): If `True`, plot the dataset region border. Defaults to `True`.
        colorbar (bool): If `True`, include a colorbar. Defaults to `True`.
        colormap (str or matplotlib.colors.Colormap): Colormap to use. Defaults to `'viridis'`.
        clim (tuple of float): Range of the colorbar. Defaults to `None`.
        clabel (str): Label of the colorbar. Defaults to `None`.
        alpha (float): Transparency level. Defaults to `None`.
        alpha_exp (float): Exponent for the alpha function (recommended between `0.4` and `1`).
            Defaults to `0`.
        **kwargs: Additional keyword arguments to customize the plot:

            - **colorbar_labelsize** (`int`): Font size for the colorbar label.
            - **colorbar_ticksize** (`int`): Font size for the colorbar ticks.
            - **figsize** (`tuple`): The size of the figure.
            - **region_color** (`str`): Color for the region border.
            - **title** (`str`): Title of the plot.
            - **title_fontsize** (`int`): Font size of the plot title.

    Returns:
        matplotlib.axes.Axes: Matplotlib axes object.
    """

    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_gridded_dataset.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    # Get spatial information for plotting
    extent = extent or _calculate_spatial_extent(region, set_global, plot_region)
    # Instantiate GeoAxes object
    ax = ax or _create_geo_axes(plot_args["figsize"], extent, projection, set_global)

    # chain basemap
    ax = plot_basemap(basemap, extent, ax=ax, set_global=set_global, show=False, **plot_args)

    # Define colormap and alpha transparency
    colormap, alpha = _get_colormap(colormap, alpha_exp, alpha)

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
        cax = ax.get_figure().add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.025,
                ax.get_position().height,
            ],
            label="Colorbar",
        )
        cbar = ax.get_figure().colorbar(im, ax=ax, cax=cax)
        cbar.set_label(clabel, fontsize=plot_args["colorbar_labelsize"])
        cbar.ax.tick_params(labelsize=plot_args["colorbar_ticksize"])
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
# Single Result plots
#####################
def plot_test_distribution(
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
        evaluation_result (EvaluationResult): Object containing test
            distributions and observed statistics.
        bins (str, int, or list): Binning strategy for the histogram. See
            :func:`matplotlib.pyplot.hist` for details on this parameter. Defaults to `'fd'`.
        percentile (int): Percentile for shading regions. Defaults to `95`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure
            and axes. Defaults to `None`.
        auto_annotate (bool or dict): If `True`, automatically formats the plot details based
            on the evaluation result. It can be further customized by passing the keyword
            arguments `xlabel`, `ylabel`, `annotation_text`, `annotation_xy`, and
            `annotation_fontsize`. Defaults to `True`.
        sim_label (str): Label for the simulated data. Defaults to `'Simulated'`.
        obs_label (str): Label for the observation data. Defaults to `'Observation'`.
        legend (bool): Whether to display the legend. Defaults to `True`.
        show (bool): If `True`, shows the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments for plot customization.

            - **color** (`str`): Color of the histogram bars.
            - **alpha** (`float`): Transparency level for the histogram bars.
            - **figsize** (`tuple`): The size of the figure.
            - **xlim** (`tuple`): Limits for the X-axis.
            - **grid** (`bool`): Whether to display grid lines. Defaults to `True`.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **xlabel**: Label of the X-axis. If `auto_annotate` is `True`, will be set to the
              test statistic name.
            - **ylabel**: Label of the Y-axis.
            - **annotate_text**: Annotate the plot. If `auto_annotate` is `True`, it will
              provide information about the statistics of the test.
            - **annotate_xy**: Position for `annotate_text` in axes fraction. Can be override
              if `auto_annotate` does not give an optimal position
            - **annotate_fontsize**: Size of the annotation.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: Matplotlib axes handle.
    """

    # Initialize plot]
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
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
    elif isinstance(observation, (list, numpy.ndarray)):
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
    Plots a calibration test (Quantile-Quantile plot) with confidence intervals.

    Args:
        evaluation_result (EvaluationResult): The evaluation result object containing
            the test distribution.
        percentile (float): Percentile to build confidence interval. Defaults to `95`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure.
            Defaults to `None`.
        label (str): Label for the plotted data. If `None`, uses
            `evaluation_result.sim_name`. Defaults to `None`.
        show (bool): If `True`, displays the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments for customizing the plot:

            - **color** (`str`): Color of the plot line and markers.
            - **marker** (`str`): Marker style for the data points.
            - **markersize** (`float`): Size of the markers.
            - **grid** (`bool`): Whether to display grid lines. Defaults to `True`.
            - **title** (`str`): Title of the plot.
            - **title_fontsize** (`int`): Font size for the plot title.
            - **xlabel** (`str`): Label for the X-axis.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object containing the plot.
    """
    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_calibration_test.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Set up QQ plots and KS test
    n = len(evaluation_result.test_distribution)
    k = numpy.arange(1, n + 1)
    # Plotting points for uniform quantiles
    pp = k / (n + 1)
    # Compute confidence intervals for order statistics using beta distribution
    inf = (100 - percentile) / 2
    sup = 100 - (100 - percentile) / 2
    ulow = beta.ppf(inf / 100, k, n - k + 1)
    uhigh = beta.ppf(sup / 100, k, n - k + 1)

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
def _plot_comparison_test(
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
        results_t (list of EvaluationResult): List of T-Test results.
        results_w (list of EvaluationResult, optional): List of W-Test results. If
            provided, they are plotted alongside the T-Test results. Defaults to `None`.
        percentile (int): Percentile for coloring W-Test results. Defaults to `95`.
        ax (matplotlib.axes.Axes): Matplotlib axes object to plot on. If `None`, a new
            figure and axes are created. Defaults to `None`.
        show (bool): If `True`, the plot is displayed after creation. Defaults to `False`.
        **kwargs: Additional keyword arguments for customizing the plot:

            - **linewidth** (`float`): Width of the error bars.
            - **capsize** (`float`): Size of the caps on the error bars.
            - **markersize** (`float`): Size of the markers.
            - **xlabel_fontsize** (`int`): Font size for the X-axis labels.
            - **ylabel** (`str`): Label for the Y-axis. Defaults to
              `'Information gain per earthquake'`.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **title** (`str`): Title of the plot. Defaults to the name of the first T-Test
              result.
            - **ylim** (`tuple`): Limits for the Y-axis.
            - **grid** (`bool`): Whether to display grid lines. Defaults to `True`.
            - **hbars** (`bool`): Whether to include horizontal bars for visual separation.
            - **legend** (`bool`): Whether to display a legend. Defaults to `True`.
            - **legend_fontsize** (`int`): Font size for the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure. Defaults
              to `True`.

    Returns:
        matplotlib.axes.Axes: The Matplotlib axes object containing the plot.
    """

    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_comparison_test.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
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
            Line2D([0], [0], color="red", lw=2,
                   label=f"T-test rejected ($\\alpha = {results_t[0].quantile[-1]}$)"),
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


def _plot_consistency_test(
    eval_results: Union[List["EvaluationResult"], "EvaluationResult"],
    normalize: bool = False,
    one_sided_lower: bool = False,
    percentile: float = 95,
    ax: Optional[pyplot.Axes] = None,
    plot_mean: bool = False,
    color: str = "black",
    show: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots the results from multiple consistency tests. The distribution of score results from
    multiple realizations of a model are plotted as a line representing a given percentile.
    The score of the observation under a model is plotted as a marker. The model is assumed
    inconsistent when the observation score lies outside the model distribution for a
    two-sided test, or lies to the right of the distribution for a one-sided test.

    Args:
        eval_results (list of EvaluationResult or EvaluationResult): Evaluation results from one
             or multiple models.
        normalize (bool): Normalize the forecast likelihood by observed likelihood. Defaults
            to `False`.
        one_sided_lower (bool): Plot for a one-sided test. Defaults to `False`.
        percentile (float): Percentile for the extent of the model score distribution. Defaults
            to `95`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure.
            Defaults to `None`.
        plot_mean (bool): Plot the mean of the test distribution. Defaults to `False`.
        color (str): Color for the line representing a model score distribution. Defaults to
            `'black'`.
        show (bool): If `True`, displays the plot. Defaults to `False`.
        **kwargs: Additional keyword arguments for plot customization:

            - **figsize** (`tuple`): The size of the figure.
            - **capsize** (`float`): Size of the caps on the error bars.
            - **linewidth** (`float`): Width of the error bars and lines.
            - **xlabel** (`str`): Label for the X-axis.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis labels.
            - **xticks_fontsize** (`int`): Font size for the X-axis ticks.
            - **title** (`str`): Title of the plot.
            - **title_fontsize** (`int`): Font size of the plot title.
            - **hbars** (`bool`): Whether to include horizontal bars for visual separation.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: Matplotlib axes object with the consistency test plot.
    """

    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_consistency_test.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # Ensure eval_results is a list
    results = list(eval_results) if isinstance(eval_results, list) else [eval_results]
    results.reverse()

    xlims = []

    for index, res in enumerate(results):
        plow, phigh, mean, observed_statistic = _process_stat_distribution(
            res, percentile, normalize, one_sided_lower
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
def _plot_concentration_ROC_diagram(
    forecast: "GriddedForecast",
    catalog: "CSEPCatalog",
    linear: bool = True,
    plot_uniform: bool = True,
    show: bool = True,
    ax: Optional[pyplot.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots the Concentration ROC Diagram for a given forecast and observed catalog.

    Args:
        forecast (GriddedForecast): Forecast object containing spatial forecast data.
        catalog (CSEPCatalog): Catalog object containing observed data.
        linear (bool): If True, uses a linear scale for the X-axis, otherwise logarithmic.
            Defaults to `True`.
        plot_uniform (bool): If True, plots the uniform (random) model as a reference.
            Defaults to `True`.
        show (bool): If True, displays the plot. Defaults to `True`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure.
            Defaults to `None`.
        **kwargs: Additional keyword arguments for customization:

            - **figsize** (`tuple`): The size of the figure.
            - **forecast_label** (`str`): Label for the forecast data in the plot.
            - **observation_label** (`str`): Label for the observation data in the plot.
            - **color** (`str`): Color for the observed data line.
            - **secondary_color** (`str`): Color for the forecast data line.
            - **linestyle** (`str`): Line style for the observed data line.
            - **title** (`str`): Title of the plot.
            - **title_fontsize** (`int`): Font size of the plot title.
            - **xlabel** (`str`): Label for the X-axis.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **grid** (`bool`): Whether to show grid lines. Defaults to `True`.
            - **legend** (`bool`): Whether to display a legend. Defaults to `True`.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **legend_framealpha** (`float`): Transparency level for the legend frame.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """

    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_concentration_ROC_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    if not catalog.region == forecast.region:
        raise AttributeError("catalog region and forecast region must be identical.")

    # Getting data
    forecast_label = plot_args.get("forecast_label", forecast.name or "Forecast")
    observed_label = plot_args.get("observation_label", "Observations")

    area_km2 = catalog.region.get_cell_area()
    obs_counts = catalog.spatial_counts()
    rate = forecast.spatial_counts()

    indices = numpy.argsort(rate)
    indices = numpy.flip(indices)

    fore_norm_sorted = numpy.cumsum(rate[indices]) / numpy.sum(rate)
    area_norm_sorted = numpy.cumsum(area_km2[indices]) / numpy.sum(area_km2)
    obs_norm_sorted = numpy.cumsum(obs_counts[indices]) / numpy.sum(obs_counts)

    # Plot data
    if plot_uniform:
        ax.plot(area_norm_sorted, area_norm_sorted, "k--", label="Uniform")

    ax.plot(
        area_norm_sorted,
        fore_norm_sorted,
        label=forecast_label,
        color=plot_args["secondary_color"],
    )

    ax.step(
        area_norm_sorted,
        obs_norm_sorted,
        label=observed_label,
        color=plot_args["color"],
        linestyle=plot_args["linestyle"],
    )

    # Plot formatting
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])
    ax.grid(plot_args["grid"])
    if not linear:
        ax.set_xscale("log")
    ax.set_ylabel(
        plot_args["ylabel"] or "True Positive Rate", fontsize=plot_args["ylabel_fontsize"]
    )
    ax.set_xlabel(
        plot_args["xlabel"] or "False Positive Rate (Normalized Area)",
        fontsize=plot_args["xlabel_fontsize"],
    )
    if plot_args["legend"]:
        ax.legend(
            loc=plot_args["legend_loc"],
            shadow=True,
            fontsize=plot_args["legend_fontsize"],
            framealpha=plot_args["legend_framealpha"],
        )
    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()

    return ax


def _plot_ROC_diagram(
    forecast: "GriddedForecast",
    catalog: "CSEPCatalog",
    linear: bool = True,
    plot_uniform: bool = True,
    show: bool = True,
    ax: Optional[pyplot.Axes] = None,
    **kwargs,
) -> matplotlib.pyplot.Axes:
    """
    Plots the ROC (Receiver Operating Characteristic) curve for a given forecast and observed
    catalog.

    Args:
        forecast (GriddedForecast): Forecast object containing spatial forecast data.
        catalog (CSEPCatalog): Catalog object containing observed data.
        linear (bool): If True, uses a linear scale for the X-axis, otherwise logarithmic.
            Defaults to `True`.
        plot_uniform (bool): If True, plots the uniform (random) model as a reference.
            Defaults to `True`.
        show (bool): If True, displays the plot. Defaults to `True`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure.
            Defaults to `None`.
        **kwargs: Additional keyword arguments for customization:

            - **figsize** (`tuple`): The size of the figure.
            - **forecast_label** (`str`): Label for the forecast data in the plot.
            - **color** (`str`): Color for the ROC curve line.
            - **linestyle** (`str`): Line style for the ROC curve.
            - **xlabel** (`str`): Label for the X-axis.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **xticks_fontsize** (`int`): Font size for the X-axis ticks.
            - **yticks_fontsize** (`int`): Font size for the Y-axis ticks.
            - **legend** (`bool`): Whether to display a legend. Defaults to `True`.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.pyplot.Axes: The Axes object with the plot.
    """

    # Initialize plot
    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_ROC_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    if not catalog.region == forecast.region:
        raise RuntimeError("catalog region and forecast region must be identical.")

    rate = forecast.spatial_counts()
    obs_counts = catalog.spatial_counts()

    indices = numpy.argsort(rate)[::-1]  # Sort in descending order

    thresholds = (rate[indices]) / numpy.sum(rate)
    obs_counts = obs_counts[indices]

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
        label=plot_args.get("forecast_label", forecast.name or "Forecast"),
        color=plot_args["color"],
        linestyle=plot_args["linestyle"],
    )

    if plot_uniform:
        ax.plot(
            numpy.arange(0, 1.001, 0.001),
            numpy.arange(0, 1.001, 0.001),
            linestyle="--",
            color="gray",
            label="Uniform",
        )

    # Plot formatting
    ax.set_ylabel(plot_args["ylabel"] or "Hit Rate", fontsize=plot_args["ylabel_fontsize"])
    ax.set_xlabel(
        plot_args["xlabel"] or "Fraction of False Alarms", fontsize=plot_args["xlabel_fontsize"]
    )
    if not linear:
        ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.tick_params(axis="x", labelsize=plot_args["xticks_fontsize"])
    ax.tick_params(axis="y", labelsize=plot_args["yticks_fontsize"])
    if plot_args["legend"]:
        ax.legend(
            loc=plot_args["legend_loc"], shadow=True, fontsize=plot_args["legend_fontsize"]
        )
    ax.set_title(plot_args["title"], fontsize=plot_args["title_fontsize"])
    if plot_args["tight_layout"]:
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax


def _plot_Molchan_diagram(
    forecast: "GriddedForecast",
    catalog: "CSEPCatalog",
    linear: bool = True,
    plot_uniform: bool = True,
    show: bool = True,
    ax: Optional[pyplot.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:

    """
    Plot the Molchan Diagram based on forecast and test catalogs using the contingency table.
    The Area Skill score and its error are shown in the legend.

    The Molchan diagram is computed following this procedure:

    1. Obtain spatial rates from the GriddedForecast and the observed events from the catalog.
    2. Rank the rates in descending order (highest rates first).
    3. Sort forecasted rates by ordering found in (2), and normalize rates so their sum is equal
       to unity.
    4. Obtain binned spatial rates from the observed catalog.
    5. Sort gridded observed rates by ordering found in (2).
    6. Test each ordered and normalized forecasted rate defined in (3) as a threshold value to
       obtain the corresponding contingency table.
    7. Define the "nu" (Miss rate) and "tau" (Fraction of spatial alarmed cells) for each
       threshold using the information provided by the corresponding contingency table defined
       in (6).

    Note:
    1. The testing catalog and forecast should have exactly the same time-window (duration).
    2. Forecasts should be defined over the same region.
    3. If calling this function multiple times, update the color in the arguments.
    4. The user can choose the x-scale (linear or log).

    Args:
        forecast (GriddedForecast): The forecast object.
        catalog (CSEPCatalog): The evaluation catalog.
        linear (bool): If True, a linear x-axis is used; if False, a logarithmic x-axis is used.
            Defaults to `True`.
        plot_uniform (bool): If True, include a uniform forecast on the plot. Defaults to
            `True`.
        show (bool): If True, displays the plot. Defaults to `True`.
        ax (matplotlib.axes.Axes): Axes object to plot on. If `None`, creates a new figure.
            Defaults to `None`.
        **kwargs: Additional keyword arguments for customization:

            - **figsize** (`tuple`): The size of the figure.
            - **forecast_label** (`str`): Label for the forecast data in the plot.
            - **color** (`str`): Color for the Molchan diagram line.
            - **linestyle** (`str`): Line style for the Molchan diagram line.
            - **xlabel** (`str`): Label for the X-axis.
            - **xlabel_fontsize** (`int`): Font size for the X-axis label.
            - **ylabel** (`str`): Label for the Y-axis.
            - **ylabel_fontsize** (`int`): Font size for the Y-axis label.
            - **legend_loc** (`str`): Location of the legend. Defaults to `'best'`.
            - **legend_fontsize** (`int`): Font size of the legend text.
            - **tight_layout** (`bool`): Whether to use tight layout for the figure.
              Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.

    Raises:
        RuntimeError: If the catalog and forecast do not have the same region.
    """

    if "plot_args" in kwargs:
        warnings.warn(
            "'plot_args' usage is deprecated and will be removed in version 1.0.\n"
            "Fine-tuning of plot appearance may not behave as expected'.\n"
            "Please use explicit arguments instead (e.g., color='red').\n"
            "Refer to the function's documentation for supported keyword arguments:\n"
            " https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_Molchan_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )
    plot_args = {**DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = pyplot.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    if not catalog.region == forecast.region:
        raise RuntimeError("Catalog region and forecast region must be identical.")

    forecast_label = plot_args.get("forecast_label", forecast.name or "Forecast")

    # Obtain forecast rates (or counts) and observed catalog aggregated in spatial cells
    rate = forecast.spatial_counts()
    obs_counts = catalog.spatial_counts()

    # Get index of rates (descending sort)
    indices = numpy.argsort(rate)
    indices = numpy.flip(indices)

    # Order forecast and cells rates by highest rate cells first
    thresholds = (rate[indices]) / numpy.sum(rate)
    obs_counts = obs_counts[indices]

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

    bin_size = 0.01
    devstd = numpy.sqrt(1 / (12 * Table_molchan["Obs_active_bins"].iloc[0]))
    devstd = devstd * bin_size**-1
    devstd = numpy.ceil(devstd + 0.5)
    devstd = devstd / bin_size**-1
    dev_std = numpy.round(devstd, 2)

    # Plot the Molchan trajectory
    ax.plot(
        Table_molchan["tau"],
        Table_molchan["nu"],
        label=f"{forecast_label}, ASS={ASscore}±{dev_std} ",
        color=plot_args["color"],
        linestyle=plot_args["linestyle"],
    )

    # Plot uniform forecast
    if plot_uniform:
        x_uniform = numpy.arange(0, 1.001, 0.001)
        y_uniform = numpy.arange(1.00, -0.001, -0.001)
        ax.plot(x_uniform, y_uniform, linestyle="--", color="gray", label="Uniform")

    # Plot formatting
    ax.set_ylabel(plot_args["ylabel"] or "Miss Rate", fontsize=plot_args["ylabel_fontsize"])
    ax.set_xlabel(
        plot_args["xlabel"] or "Fraction of area occupied by alarms",
        fontsize=plot_args["xlabel_fontsize"],
    )
    if not linear:
        ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=plot_args["xlabel_fontsize"])
    ax.tick_params(axis="y", labelsize=plot_args["ylabel_fontsize"])
    ax.legend(loc=plot_args["legend_loc"], shadow=True, fontsize=plot_args["legend_fontsize"])
    ax.set_title(plot_args["title"] or "Molchan Diagram", fontsize=plot_args["title_fontsize"])

    if plot_args["tight_layout"]:
        fig.tight_layout()
    if show:
        pyplot.show()
    return ax


#####################
# Plot helper functions
#####################
def _get_marker_style(obs_stat: float, p: Sequence[float], one_sided_lower: bool) -> str:
    """
    Returns the matplotlib marker style as a format string.

    Args:
        obs_stat (float): The observed statistic.
        p (Sequence[float, float]): A tuple of lower and upper percentiles.
        one_sided_lower (bool): Indicates if the test is one-sided lower.

    Returns:
        str: A format string representing the marker style.
    """
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


def _get_marker_t_color(distribution: Sequence[float]) -> str:
    """
    Returns the color for the marker based on the distribution.

    Args:
        distribution (Sequence[float, float]): A tuple representing the lower and upper bounds
                                            of the test distribution.

    Returns:
        str: Marker color
    """
    if distribution[0] > 0.0 and distribution[1] > 0.0:
        color = "green"
    elif distribution[0] < 0.0 and distribution[1] < 0.0:
        color = "red"
    else:
        color = "grey"

    return color


def _get_marker_w_color(distribution: float, percentile: float) -> bool:
    """
    Returns a boolean indicating whether the distribution's percentile is below a given
    threshold.

    Args:
        distribution (float): The value of the distribution's percentile.
        percentile (float): The percentile threshold.

    Returns:
        bool: True if the distribution's percentile is below the threshold, False otherwise.
    """
    if distribution < (1 - percentile / 100):
        fmt = True
    else:
        fmt = False

    return fmt


def _get_axis_limits(points: Union[Sequence, numpy.ndarray],
                     border: float = 0.05) -> Tuple[float, float]:
    """
    Returns a tuple of x_min and x_max given points on a plot.

    Args:
        points (numpy.ndarray): An array of points.
        border (float): The border fraction to apply to the limits.

    Returns:
        Sequence[float, float]: The x_min and x_max values adjusted with the border.
    """
    x_min = numpy.min(points)
    x_max = numpy.max(points)
    xd = (x_max - x_min) * border
    return x_min - xd, x_max + xd


def _get_basemap(basemap: str) -> Union[img_tiles.GoogleTiles, DatasetReader]:
    """
    Returns the basemap tiles for a given basemap type or web service.

    Args:
        basemap (str): The type of basemap for cartopy, an URL for a web service or a TIF file
                       path.

    Returns:
        Union[img_tiles.GoogleTiles, rasterio.io.DatasetReader]: The corresponding tiles or
                raster object.

    """
    last_cache = os.path.join(
        os.path.dirname(cartopy.config["cache_dir"]), "last_cartopy_cache"
    )

    def _clean_cache(basemap_):
        if os.path.isfile(last_cache):
            with open(last_cache, "r") as fp:
                cache_src = fp.read()
            if cache_src != basemap_:
                if os.path.isdir(cartopy.config["cache_dir"]):
                    print(f"Cleaning existing {basemap_} cache")
                    shutil.rmtree(cartopy.config["cache_dir"])

    def _save_cache_src(basemap_):
        with open(last_cache, "w") as fp:
            fp.write(basemap_)

    cache = True

    warning_message_to_suppress = (
        "Cartopy created the following directory to cache" " GoogleWTS tiles"
    )
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
            return rio_open(basemap)

        else:
            try:
                _clean_cache(basemap)
                webservice = basemap
                tiles = img_tiles.GoogleTiles(url=webservice, cache=cache)
                _save_cache_src(basemap)
            except Exception as e:
                raise ValueError(f"Basemap type not valid or not implemented. {e}")

    return tiles


def _autosize_scatter(
    values: numpy.ndarray,
    min_size: float = 50.0,
    max_size: float = 400.0,
    power: float = 3.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> numpy.ndarray:
    """
    Auto-sizes scatter plot markers based on values.

    Args:
        values (numpy.ndarray): The data values (e.g., magnitude) to base the sizing on.
        min_size (float): The minimum marker size.
        max_size (float): The maximum marker size.
        power (float): The power to apply for scaling.
        min_val (Optional[float]): The minimum value (e.g., magnitude) for normalization.
        max_val (Optional[float]): The maximum value (e.g., magnitude) for normalization.

    Returns:
        numpy.ndarray: The calculated marker sizes.
    """
    min_val = min_val or numpy.min(values)
    max_val = max_val or numpy.max(values)
    normalized_values = ((values - min_val) / (max_val - min_val)) ** power
    marker_sizes = min_size + normalized_values * (max_size - min_size) * bool(power)
    return marker_sizes


def _autoscale_histogram(
    ax: matplotlib.axes.Axes,
    bin_edges: numpy.ndarray,
    simulated: numpy.ndarray,
    observation: numpy.ndarray,
    mass: float = 99.5,
) -> matplotlib.axes.Axes:
    """
    Autoscale the histogram axes based on the data distribution.

    Args:
        ax (matplotlib.axes.Axes): The axes to apply the scaling to.
        bin_edges (numpy.ndarray): The edges of the histogram bins.
        simulated (numpy.ndarray): Simulated data values.
        observation (numpy.ndarray): Observed data values.
        mass (float): The percentage of the data mass to consider.

    Returns:
        matplotlib.axes.Axes: The scaled axes
    """

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
    ax: matplotlib.axes.Axes,
    evaluation_result: "EvaluationResult",
    auto_annotate: bool,
    plot_args: Dict[str, Any],
) -> matplotlib.axes.Axes:
    """
    Annotates a distribution plot based on the evaluation result type.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        evaluation_result (EvaluationResult): The evaluation result object.
        auto_annotate (bool): If True, automatically annotates the plot based on result type.
        plot_args (Dict[str, Any]): Additional plotting arguments.

    Returns:
        matplotlib.axes.Axes: The annotated axes.
    """
    annotation_text = None
    annotation_xy = None
    title = None
    xlabel = None
    ylabel = None

    if auto_annotate:
        if evaluation_result.name == "Catalog N-Test":
            xlabel = "Event Counts"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.5, 0.3)
            if isinstance(evaluation_result.quantile, (list, tuple, numpy.ndarray)):
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
            xlabel = "Spatial Statistic"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.2, 0.6)
            annotation_text = (
                f"$\\gamma = P(X \\leq x) = "
                f"{numpy.array(evaluation_result.quantile).ravel()[-1]:.2f}$\n"
                f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
            )

        elif evaluation_result.name == "Catalog M-Test":
            xlabel = "Magnitude Statistic"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.55, 0.6)
            annotation_text = (
                f"$\\gamma = P(X \\geq x) = "
                f"{numpy.array(evaluation_result.quantile).ravel()[0]:.2f}$\n"
                f"$\\omega = {evaluation_result.observed_statistic:.2f}$"
            )
        elif evaluation_result.name == "Catalog PL-Test":
            xlabel = "Pseudo-Likelihood"
            ylabel = "Number of Catalogs"
            title = f"{evaluation_result.name}: {evaluation_result.sim_name}"
            annotation_xy = (0.55, 0.3)
            annotation_text = (
                f"$\\gamma = P(X \\leq x) = "
                f"{numpy.array(evaluation_result.quantile).ravel()[-1]:.2f}$\n"
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


def _calculate_spatial_extent(
    element: Union["CSEPCatalog", "CartesianGrid2D"],
    set_global: bool,
    region_border: bool,
    padding_fraction: float = 0.05,
) -> Optional[List[float]]:
    """
    Calculates the spatial extent for plotting based on the catalog.

    Args:
        element (CSEPCatalog), CartesianGrid2D: The catalog or region object to base the extent
                                                on.
        set_global (bool): If True, sets the extent to the global view.
        region_border (bool): If True, uses the catalog's region border.
        padding_fraction (float): The fraction of padding to apply to the extent.

    Returns:
        Optional[List[float]]: The calculated extent or None if global view is set.
    """
    # todo: perhaps calculate extent also from chained ax object
    bbox = element.get_bbox()
    if region_border:
        try:
            bbox = element.region.get_bbox()
        except AttributeError:
            pass

    if set_global:
        return None

    dh = (bbox[1] - bbox[0]) * padding_fraction
    dv = (bbox[3] - bbox[2]) * padding_fraction
    return [bbox[0] - dh, bbox[1] + dh, bbox[2] - dv, bbox[3] + dv]


def _create_geo_axes(
    figsize: Optional[Tuple[float, float]],
    extent: Optional[List[float]],
    projection: Union[ccrs.Projection, str],
    set_global: bool,
) -> pyplot.Axes:
    """
    Creates and returns GeoAxes for plotting.

    Args:
        figsize (Optional[Tuple[float, float]]): The size of the figure.
        extent (Optional[List[float]]): The spatial extent to set.
        projection (Union[ccrs.Projection, str]): The projection to use.
        set_global (bool): If True, sets the global view.

    Returns:
        pyplot.Axes: The created GeoAxes object.
    """

    if projection == "approx":
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        central_latitude = (extent[2] + extent[3]) / 2.0
        # Set plot aspect according to local longitude-latitude ratio in metric units
        LATKM = 110.574  # length of a ° of latitude [km] --> ignores Earth's flattening
        ax.set_aspect(LATKM / (111.320 * numpy.cos(numpy.deg2rad(central_latitude))))
    elif projection is None:
        projection = ccrs.PlateCarree()
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    else:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    if set_global:
        ax.set_global()
    elif extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def _add_gridlines(ax: matplotlib.axes.Axes, grid_labels: bool, grid_fontsize: float) -> None:
    """
    Adds gridlines and optionally labels to the axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to add gridlines to.
        grid_labels (bool): If True, labels the gridlines.
        grid_fontsize (float): The font size of the grid labels.
    """
    gl = ax.gridlines(draw_labels=grid_labels, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style["fontsize"] = grid_fontsize
    gl.ylabel_style["fontsize"] = grid_fontsize
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def _get_colormap(
    cmap: Union[str, matplotlib.colors.Colormap],
    alpha_exp: float,
    alpha_0: Optional[float] = None,
) -> Tuple[matplotlib.colors.ListedColormap, Optional[float]]:
    """
    Defines the colormap and applies alpha transparency based on the given parameters.

    Args:
        cmap (Union[str, matplotlib.colors.Colormap]): The colormap to use.
        alpha_exp (float): The exponent to control transparency scaling.
        alpha_0 (Optional[float]): If set, applies a uniform alpha across the colormap.

    Returns:
        Tuple[matplotlib.colors.ListedColormap, Optional[float]]: The modified colormap
        and the alpha value used for the entire colormap.
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


def _process_stat_distribution(
    res: "EvaluationResult",
    percentile: float,
    normalize: bool,
    one_sided_lower: bool,
) -> Tuple[float, float, float, float]:
    """
    Processes the statistical distribution based on its type and returns plotting values.

    Args:
        res (EvaluationResult): The evaluation result object containing the distribution data.
        percentile (float): The percentile for calculating the confidence intervals.
        variance (Optional[float]): The variance of the negative binomial distribution, if
                                    applicable.
        normalize (bool): If True, normalizes the distribution by the observed statistic.
        one_sided_lower (bool): If True, performs a one-sided lower test.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the lower percentile,
        upper percentile, mean, and observed statistic.
    """
    dist_type = res.test_distribution[0]

    if dist_type == "poisson":
        mean = res.test_distribution[1]
        plow = poisson.ppf((1 - percentile / 100.0) / 2.0, mean)
        phigh = poisson.ppf(1 - (1 - percentile / 100.0) / 2.0, mean)
        observed_statistic = res.observed_statistic
    elif dist_type == "negative_binomial":
        mean = res.test_distribution[1]
        variance = res.test_distribution[2]
        upsilon = 1.0 - ((variance - mean) / variance)
        tau = mean**2 / (variance - mean)
        plow = nbinom.ppf((1 - percentile / 100.0) / 2.0, tau, upsilon)
        phigh = nbinom.ppf(1 - (1 - percentile / 100.0) / 2.0, tau, upsilon)
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


# Public export of function wrappers for backwards/legacy compatibility.
from .plots_legacy import (plot_cumulative_events_versus_time,
                           plot_cumulative_events_versus_time_dev,
                           plot_histogram,
                           plot_ecdf,
                           plot_magnitude_histogram_dev,
                           plot_basemap,
                           plot_spatial_dataset,
                           plot_number_test,
                           plot_magnitude_test,
                           plot_distribution_test,
                           plot_likelihood_test,
                           plot_spatial_test,
                           plot_poisson_consistency_test,
                           plot_comparison_test,
                           plot_consistency_test,
                           plot_pvalues_and_intervals,
                           plot_concentration_ROC_diagram,
                           plot_ROC_diagram,
                           plot_Molchan_diagram)

