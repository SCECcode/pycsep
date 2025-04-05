import time
import warnings
from functools import wraps

# Third-party imports
import numpy
import string
import scipy.stats
import matplotlib
import matplotlib.lines
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as pyplot
import cartopy.crs as ccrs
from cartopy.io import img_tiles
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# PyCSEP imports
from csep.utils.calc import bin1d_vec
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.utils.plots import (_plot_cumulative_events_versus_time, _plot_basemap,
                              _plot_comparison_test, _plot_consistency_test, _plot_concentration_ROC_diagram, _plot_ROC_diagram, _plot_Molchan_diagram)


def plot_cumulative_events_versus_time_dev(xdata, ydata, obs_data,
                                           plot_args, show=False):
    """

    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead.


    Args:
        xdata (ndarray): time bins for plotting shape (N,)
        ydata (ndarray or list like): ydata for plotting; shape (N,5) in order 2.5%Per, 25%Per, 50%Per, 75%Per, 97.5%Per
        obs_data (ndarry): same shape as xdata
        plot_args:
        show:

    Returns:

    """
    warnings.warn(
        "'plot_cumulative_events_versus_time_dev' is deprecated and will be removed in version 1.0.\n"
        "Please use 'plot_cumulative_events_versus_time' with the appropriate keyword arguments instead:\n"
        "    catalog_forecast (CatalogForecast), observation (CSEPCatalog), show (bool), etc.\n\n"
        "See the updated documentation at:\n"
        "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_cumulative_events_versus_time.html\n",
        DeprecationWarning,
        stacklevel=2
    )

    figsize = plot_args.get('figsize', None)
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    legend_loc = plot_args.get('legend_loc', 'best')
    title = plot_args.get('title', 'Cumulative Event Counts')
    xlabel = plot_args.get('xlabel', 'Days')

    fig, ax = pyplot.subplots(figsize=figsize)
    try:
        fifth_per = ydata[0, :]
        first_quar = ydata[1, :]
        med_counts = ydata[2, :]
        second_quar = ydata[3, :]
        nine_fifth = ydata[4, :]
    except:
        raise TypeError("ydata must be a [N,5] ndarray.")
    # plotting

    ax.plot(xdata, obs_data, color='black', label=obs_label)
    ax.plot(xdata, med_counts, color='red', label=sim_label)
    ax.fill_between(xdata, fifth_per, nine_fifth, color='red', alpha=0.2,
                    label='5%-95%')
    ax.fill_between(xdata, first_quar, second_quar, color='red', alpha=0.5,
                    label='25%-75%')
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative event count')
    ax.set_title(title)
    # pyplot.subplots_adjust(right=0.75)
    # annotate the plot with information from data
    # ax.annotate(str(observation), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    # save figure
    filename = plot_args.get('filename', None)
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)
    # optionally show figure
    if show:
        pyplot.show()

    return ax


@wraps(_plot_cumulative_events_versus_time)
def plot_cumulative_events_versus_time(*args, **kwargs):
    """
    Legacy-compatible wrapper for plot_cumulative_events_versus_time.

    Deprecated usage will emit warnings and fall back to legacy logic.
    This wrapper will be removed in a future version.
    """
    from csep.core.forecasts import CatalogForecast
    from csep.core.catalogs import CSEPCatalog
    from .plots_legacy import _plot_cumulative_events_versus_time_legacy_impl

    try:
        return _plot_cumulative_events_versus_time(*args, **kwargs)
    except Exception as e:
        # Determine if the call is legacy-style
        is_legacy_call = False

        if len(args) >= 2:
            is_legacy_call = (
                    not isinstance(args[0], CatalogForecast) and
                    isinstance(args[1], CSEPCatalog)
            )
        elif 'catalog_forecast' not in kwargs or not isinstance(kwargs['catalog_forecast'],
                                                                CatalogForecast):
            is_legacy_call = True

        if is_legacy_call:
            warnings.warn(
            "'plot_cumulative_events_versus_time' was called with deprecated arguments, and falling back to legacy implementation.\n"
            "As of version 1.0, this behavior is deprecated. "
            "Please use the new keyword arguments:\n"
            "    catalog_forecast (CatalogForecast), observation (CSEPCatalog), show (bool), etc.\n\n"
            "See the updated documentation at:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_cumulative_events_versus_time.html\n",
            DeprecationWarning,
            stacklevel=2
        )

            return _plot_cumulative_events_versus_time_legacy_impl(*args, **kwargs)

        raise


def _plot_cumulative_events_versus_time_legacy_impl(stochastic_event_sets, observation,
                                       show=False, plot_args=None):
    """
    Same as below but performs the statistics on numpy arrays without using pandas data frames.

    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead.

    Args:
        stochastic_event_sets:
        observation:
        show:
        plot_args:

    Returns:
        ax: matplotlib.Axes
    """
    plot_args = plot_args or {}
    figsize = plot_args.get('figsize', None)
    fig, ax = pyplot.subplots(figsize=figsize)
    # get global information from stochastic event set
    t0 = time.time()
    n_cat = len(stochastic_event_sets)

    extreme_times = []
    for ses in stochastic_event_sets:
        start_epoch = datetime_to_utc_epoch(ses.start_time)
        end_epoch = datetime_to_utc_epoch(ses.end_time)
        if start_epoch == None or end_epoch == None:
            continue

        extreme_times.append((start_epoch, end_epoch))

    # offsets to start at 0 time and converts from millis to hours
    time_bins, dt = numpy.linspace(numpy.min(extreme_times),
                                   numpy.max(extreme_times), 100,
                                   endpoint=True, retstep=True)
    n_bins = time_bins.shape[0]
    binned_counts = numpy.zeros((n_cat, n_bins))
    for i, ses in enumerate(stochastic_event_sets):
        n_events = ses.data.shape[0]
        ses_origin_time = ses.get_epoch_times()
        inds = bin1d_vec(ses_origin_time, time_bins)
        for j in range(n_events):
            binned_counts[i, inds[j]] += 1
        if (i + 1) % 1500 == 0:
            t1 = time.time()
            print(f"Processed {i + 1} catalogs in {t1 - t0} seconds.")
    t1 = time.time()
    print(f'Collected binned counts in {t1 - t0} seconds.')
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

    # update time_bins for plotting
    millis_to_hours = 60 * 60 * 1000 * 24
    time_bins = (time_bins - time_bins[0]) / millis_to_hours
    time_bins = time_bins + (dt / millis_to_hours)
    # make all arrays start at zero
    time_bins = numpy.insert(time_bins, 0, 0)
    fifth_per = numpy.insert(fifth_per, 0, 0)
    first_quar = numpy.insert(first_quar, 0, 0)
    med_counts = numpy.insert(med_counts, 0, 0)
    second_quar = numpy.insert(second_quar, 0, 0)
    nine_fifth = numpy.insert(nine_fifth, 0, 0)
    obs_summed_counts = numpy.insert(obs_summed_counts, 0, 0)

    # get values from plotting args
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    legend_loc = plot_args.get('legend_loc', 'best')
    title = plot_args.get('title', 'Cumulative Event Counts')
    # plotting
    ax.plot(time_bins, obs_summed_counts, color='black', label=obs_label)
    ax.plot(time_bins, med_counts, color='red', label=sim_label)
    ax.fill_between(time_bins, fifth_per, nine_fifth, color='red', alpha=0.2,
                    label='5%-95%')
    ax.fill_between(time_bins, first_quar, second_quar, color='red', alpha=0.5,
                    label='25%-75%')
    ax.legend(loc=legend_loc)
    ax.set_xlabel('Days since Mainshock')
    ax.set_ylabel('Cumulative Event Count')
    ax.set_title(title)
    pyplot.subplots_adjust(right=0.75)
    # annotate the plot with information from data
    # ax.annotate(str(observation), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    # save figure
    filename = plot_args.get('filename', None)
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)
    # optionally show figure
    if show:
        pyplot.show()

    return ax


def plot_histogram(simulated, observation, bins='fd', percentile=None,
                   show=False, axes=None, catalog=None, plot_args=None):
    """
    Plots histogram of single statistic for stochastic event sets and observations. The function will behave differently depending on the inumpyuts.

    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead.


    Simulated should always be either a list or numpy.array where there would be one value per data in the stochastic event
    set. Observation could either be a scalar or a numpy.array/list. If observation is a scale a vertical line would be
    plotted, if observation is iterable a second histogram would be plotted.

    This allows for comparisons to be made against catalogs where there are multiple values e.g., magnitude, and single values
    e.g., event count.

    If an axis handle is included, additional function calls will only addition extra simulations, observations will not be
    plotted. Since this function returns an axes handle, any extra modifications to the figure can be made using that.

    Args:
        simulated (numpy.arrays): numpy.array like representation of statistics computed from catalogs.
        observation(numpy.array or scalar): observation to plot against stochastic event set
        filename (str): filename to save figure
        show (bool): show interactive version of the figure
        ax (axis object): axis object with interface defined by matplotlib
        catalog (csep.AbstractBaseCatalog): used for annotating the figures
        plot_args (dict): additional plotting commands. TODO: Documentation

    Returns:
        axis: matplolib axes handle
    """
    # Plotting

    warnings.warn(
        "'plot_histogram' is deprecated and will be removed in version 1.0.\n"
        "Please use 'plot_test_distribution' instead, which provides improved support for test statistics.\n\n"
        "See the documentation at:\n"
        "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )

    plot_args = plot_args or {}
    chained = False
    figsize = plot_args.get('figsize', None)
    if axes is not None:
        chained = True
        ax = axes
    else:
        if catalog:
            fig, ax = pyplot.subplots(figsize=figsize)
        else:
            fig, ax = pyplot.subplots()

    # parse plotting arguments
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    xlabel = plot_args.get('xlabel', 'X')
    ylabel = plot_args.get('ylabel', 'Frequency')
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    title = plot_args.get('title', None)
    legend_loc = plot_args.get('legend_loc', 'best')
    legend = plot_args.get('legend', True)
    bins = plot_args.get('bins', bins)
    color = plot_args.get('color', '')
    filename = plot_args.get('filename', None)
    xlim = plot_args.get('xlim', None)

    # this could throw an error exposing bad implementation
    observation = numpy.array(observation)

    try:
        n = len(observation)
    except TypeError:
        ax.axvline(x=observation, color='black', linestyle='--',
                   label=obs_label)

    else:
        # remove any nan values
        observation = observation[~numpy.isnan(observation)]
        ax.hist(observation, bins=bins, label=obs_label, edgecolor=None,
                linewidth=0)

    # remove any potential nans from arrays
    simulated = numpy.array(simulated)
    simulated = simulated[~numpy.isnan(simulated)]

    if color:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, label=sim_label,
                                        color=color, edgecolor=None,
                                        linewidth=0)
    else:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, label=sim_label,
                                        edgecolor=None, linewidth=0)

    # color bars for rejection area
    if percentile is not None:
        inc = (100 - percentile) / 2
        inc_high = 100 - inc
        inc_low = inc
        p_high = numpy.percentile(simulated, inc_high)
        idx_high = numpy.digitize(p_high, bin_edges)
        p_low = numpy.percentile(simulated, inc_low)
        idx_low = numpy.digitize(p_low, bin_edges)

    # show 99.5% of data
    if xlim is None:
        upper_xlim = numpy.percentile(simulated, 99.75)
        upper_xlim = numpy.max([upper_xlim, numpy.max(observation)])
        d_bin = bin_edges[1] - bin_edges[0]
        upper_xlim = upper_xlim + 2 * d_bin

        lower_xlim = numpy.percentile(simulated, 0.25)
        lower_xlim = numpy.min([lower_xlim, numpy.min(observation)])
        lower_xlim = lower_xlim - 2 * d_bin

        try:
            ax.set_xlim([lower_xlim, upper_xlim])
        except ValueError:
            print('Ignoring observation in axis scaling because inf or -inf')
            upper_xlim = numpy.percentile(simulated, 99.75)
            upper_xlim = upper_xlim + 2 * d_bin

            lower_xlim = numpy.percentile(simulated, 0.25)
            lower_xlim = lower_xlim - 2 * d_bin

            ax.set_xlim([lower_xlim, upper_xlim])
    else:
        ax.set_xlim(xlim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend_loc)

    # hacky workaround for coloring legend, by calling after legend is drawn.
    if percentile is not None:
        for idx in range(idx_low):
            patches[idx].set_fc('red')
        for idx in range(idx_high, len(patches)):
            patches[idx].set_fc('red')
    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)
    if show:
        pyplot.show()
    return ax


def plot_ecdf(x, ecdf, axes=None, xv=None, show=False, plot_args=None):
    """ Plots empirical cumulative distribution function.  """
    warnings.warn(
        "'plot_ecdf' will be removed in a future version of the package.\n"
        "If you rely on this functionality, consider switching to a custom ECDF implementation",
        DeprecationWarning,
        stacklevel=2
    )

    plot_args = plot_args or {}
    # get values from plotting args
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    xlabel = plot_args.get('xlabel', 'X')
    ylabel = plot_args.get('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    legend_loc = plot_args.get('legend_loc', 'best')
    filename = plot_args.get('filename', None)

    # make figure
    if axes == None:
        fig, ax = pyplot.subplots()
    else:
        ax = axes
        fig = axes.figure
    ax.plot(x, ecdf, label=sim_label)
    if xv:
        ax.axvline(x=xv, color='black', linestyle='--', label=obs_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    # if data is not None:
    #     ax.annotate(str(data), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)

    if show:
        pyplot.show()

    return ax


def plot_magnitude_histogram_dev(ses_data, obs, plot_args, show=False):
    warnings.warn(
        "'plot_magnitude_histogram_dev' is deprecated and will be removed in version 1.0.\n"
        "Please use 'plot_magnitude_instagram' instead.\n"
        "See the documentation at:\n"
        "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_magnitude_instagram.html",
        DeprecationWarning,
        stacklevel=2
    )

    bin_edges, obs_hist = obs.magnitude_counts(retbins=True)
    n_obs = numpy.sum(obs_hist)
    event_counts = numpy.sum(ses_data, axis=1)
    # normalize all histograms by counts in each
    scale = n_obs / event_counts
    # use broadcasting
    ses_data = ses_data * scale.reshape(-1, 1)
    figsize = plot_args.get('figsize', None)
    fig = pyplot.figure(figsize=figsize)
    ax = fig.gca()
    u3etas_median = numpy.median(ses_data, axis=0)
    u3etas_low = numpy.percentile(ses_data, 2.5, axis=0)
    u3etas_high = numpy.percentile(ses_data, 97.5, axis=0)
    u3etas_min = numpy.min(ses_data, axis=0)
    u3etas_max = numpy.max(ses_data, axis=0)
    u3etas_emax = u3etas_max - u3etas_median
    u3etas_emin = u3etas_median - u3etas_min
    dmw = bin_edges[1] - bin_edges[0]
    bin_edges_plot = bin_edges + dmw / 2

    # u3etas_emax = u3etas_max
    # plot 95% range as rectangles
    rectangles = []
    for i in range(len(bin_edges)):
        width = dmw / 2
        height = u3etas_high[i] - u3etas_low[i]
        xi = bin_edges[i] + width / 2
        yi = u3etas_low[i]
        rect = matplotlib.patches.Rectangle((xi, yi), width, height)
        rectangles.append(rect)
    pc = matplotlib.collections.PatchCollection(rectangles, facecolor='blue',
                                                alpha=0.3, edgecolor='blue')
    ax.add_collection(pc)
    # plot whiskers
    sim_label = plot_args.get('sim_label', 'Simulated Catalogs')
    obs_label = plot_args.get('obs_label', 'Observed Catalog')
    xlim = plot_args.get('xlim', None)
    title = plot_args.get('title', "UCERF3-ETAS Histogram")
    filename = plot_args.get('filename', None)

    ax.errorbar(bin_edges_plot, u3etas_median, yerr=[u3etas_emin, u3etas_emax],
                xerr=0.8 * dmw / 2, fmt=' ', label=sim_label,
                color='blue', alpha=0.7)
    ax.plot(bin_edges_plot, obs_hist, '.k', markersize=10, label=obs_label)
    ax.legend(loc='upper right')
    ax.set_xlim(xlim)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Event count per magnitude bin')
    ax.set_title(title)
    # ax.annotate(str(comcat), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    # pyplot.subplots_adjust(right=0.75)
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)
    if show:
        pyplot.show()
    return ax


@wraps(_plot_basemap)
def plot_basemap(*args, **kwargs):
    legacy_keys = {"tile_scaling", "apprx", "central_latitude"}

    if any(k in kwargs for k in legacy_keys):
        warnings.warn(
            "'plot_basemap' was called with deprecated arguments. These options are no longer supported "
            "and will be removed in version 1.0.\n\n"
            "Please consult the updated documentation at:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_basemap.html\n",
            DeprecationWarning,
            stacklevel=2
        )

    return _plot_basemap(*args, **kwargs)


def _get_basemap(basemap):
    if basemap == 'stamen_terrain':
        tiles = img_tiles.Stamen('terrain')
    elif basemap == 'stamen_terrain-background':
        tiles = img_tiles.Stamen('terrain-background')
    elif basemap == 'google-satellite':
        tiles = img_tiles.GoogleTiles(style='satellite')
    elif basemap == 'ESRI_terrain':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/' \
                     'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_imagery':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/' \
                     'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_relief':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/' \
                     'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_topo':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/' \
                     'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    else:
        try:
            webservice = basemap
            tiles = img_tiles.GoogleTiles(url=webservice)
        except:
            raise ValueError('Basemap type not valid or not implemented')

    return tiles


def plot_spatial_dataset(gridded, region, ax=None, show=False, extent=None,
                         set_global=False, plot_args=None, **kwargs):
    """
    Legacy-compatible wrapper for plot_gridded_dataset.
    """

    plot_args = plot_args or {}
    warnings.warn(
        "'plot_spatial_dataset' is deprecated and will be removed in version 1.0.\n"
        "Please use 'plot_gridded_dataset' instead.\n"
        "See https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_gridded_dataset.html",
        DeprecationWarning,
        stacklevel=2
    )

    # Get spatial information for plotting
    bbox = region.get_bbox()
    if extent is None and not set_global:
        extent = [bbox[0], bbox[1], bbox[2], bbox[3]]

    # Retrieve plot arguments
    plot_args = plot_args or {}
    # figure and axes properties
    figsize = plot_args.get('figsize', None)
    title = plot_args.get('title', None)
    title_size = plot_args.get('title_size', None)
    filename = plot_args.get('filename', None)
    # cartopy properties
    projection = plot_args.get('projection',
                               ccrs.PlateCarree(central_longitude=0.0))
    grid = plot_args.get('grid', True)
    grid_labels = plot_args.get('grid_labels', False)
    grid_fontsize = plot_args.get('grid_fontsize', False)
    basemap = plot_args.get('basemap', None)
    coastline = plot_args.get('coastline', True)
    borders = plot_args.get('borders', False)
    tile_scaling = plot_args.get('tile_scaling', 'auto')
    linewidth = plot_args.get('linewidth', True)
    linecolor = plot_args.get('linecolor', 'black')
    region_border = plot_args.get('region_border', True)
    # color bar properties
    include_cbar = plot_args.get('include_cbar', True)
    cmap = plot_args.get('cmap', None)
    clim = plot_args.get('clim', None)
    clabel = plot_args.get('clabel', None)
    clabel_fontsize = plot_args.get('clabel_fontsize', None)
    cticks_fontsize = plot_args.get('cticks_fontsize', None)
    alpha = plot_args.get('alpha', 1)
    alpha_exp = plot_args.get('alpha_exp', 0)

    apprx = False
    central_latitude = 0.0
    if projection == 'fast':
        projection = ccrs.PlateCarree()
        apprx = True
        n_lats = len(region.ys) // 2
        central_latitude = region.ys[n_lats]

    # Instantiate GeoAxes object
    if ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    else:
        fig = ax.get_figure()

    if set_global:
        ax.set_global()
        region_border = False
    else:
        ax.set_extent(extents=extent,
                      crs=ccrs.PlateCarree())  # Defined extent always in lat/lon

    # Basemap plotting
    ax = plot_basemap(basemap, extent, ax=ax, coastline=coastline,
                      borders=borders,
                      linecolor=linecolor, linewidth=linewidth,
                      projection=projection, apprx=apprx,
                      central_latitude=central_latitude,
                      tile_scaling=tile_scaling, set_global=set_global)

    ## Define colormap and transparency function
    if isinstance(cmap, str) or not cmap:
        cmap = pyplot.get_cmap(cmap)
    cmap_tup = cmap(numpy.arange(cmap.N))
    if isinstance(alpha_exp, (float, int)):
        if alpha_exp != 0:
            cmap_tup[:, -1] = numpy.linspace(0, 1, cmap.N) ** alpha_exp
            alpha = None
    cmap = matplotlib.colors.ListedColormap(cmap_tup)

    ## Plot spatial dataset
    lons, lats = numpy.meshgrid(numpy.append(region.xs, bbox[1]),
                                numpy.append(region.ys, bbox[3]))
    im = ax.pcolor(lons, lats, gridded, cmap=cmap, alpha=alpha, snap=True,
                   transform=ccrs.PlateCarree())
    im.set_clim(clim)


    # Colorbar options
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    if include_cbar:
        cax = fig.add_axes(
            [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.025,
             ax.get_position().height],
            label='Colorbar')
        cbar = fig.colorbar(im, ax=ax, cax=cax)
        cbar.set_label(clabel, fontsize=clabel_fontsize)
        cbar.ax.tick_params(labelsize=cticks_fontsize)

    # Gridline options
    if grid:
        gl = ax.gridlines(draw_labels=grid_labels, alpha=0.5)
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style['fontsize'] = grid_fontsize
        gl.ylabel_style['fontsize'] = grid_fontsize
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    if region_border:
        pts = region.tight_bbox()
        ax.plot(pts[:, 0], pts[:, 1], lw=1, color='black',
                transform=ccrs.PlateCarree())

    # matplotlib figure options
    ax.set_title(title, y=1.06, fontsize=title_size)
    if filename is not None:
        ax.get_figure().savefig(filename + '.pdf')
        ax.get_figure().savefig(filename + '.png', dpi=300)
    if show:
        pyplot.show()

    return ax


def plot_number_test(evaluation_result, axes=None, show=True, plot_args=None):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the n-test.


    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead

    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult
        axes (matplotlib.Axes): axes object used to chain this plot
        show (bool): if true, call pyplot.show()
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * text_fontsize: (:class:`float`) - default: 14
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True
        * percentile (:class:`float`) Critial region to shade on histogram - default: 95
        * bins: (:class:`str`) - Set binning type. see matplotlib.hist for more info - default: 'auto'
        * xy: (:class:`list`/:class:`tuple`) - default: (0.55, 0.3)

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure

    """
    warnings.warn(
        "'plot_number_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    # chain plotting axes if requested
    if axes:
        chained = True
    else:
        chained = False

    # default plotting arguments
    plot_args = plot_args or {}
    title = plot_args.get('title', 'Number Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'Event count of catalogs')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    text_fontsize = plot_args.get('text_fontsize', 14)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.5, 0.3))

    # set default plotting arguments
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    ax = plot_histogram(evaluation_result.test_distribution,
                        evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        try:
            ax.annotate(
                '$\delta_1 = P(X \geq x) = {:.2f}$\n$\delta_2 = P(X \leq x) = {:.2f}$\n$\omega = {:d}$'
                .format(*evaluation_result.quantile,
                        evaluation_result.observed_statistic),
                xycoords='axes fraction',
                xy=xy,
                fontsize=text_fontsize)
        except:
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile,
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if tight_layout:
        ax.figure.tight_layout()

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_magnitude_test(evaluation_result, axes=None, show=True,
                        plot_args=None):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the M-test.

    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead

    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult
        axes (matplotlib.Axes): axes object used to chain this plot
        show (bool): if true, call pyplot.show()
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True
        * percentile (:class:`float`) Critial region to shade on histogram - default: 95
        * bins: (:class:`str`) - Set binning type. see matplotlib.hist for more info - default: 'auto'
        * xy: (:class:`list`/:class:`tuple`) - default: (0.55, 0.6)

    Returns:
        ax (matplotlib.Axes): containing the new plot

    """
    warnings.warn(
        "'plot_magnitude_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    plot_args = plot_args or {}
    title = plot_args.get('title', 'Magnitude Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'D* Statistic')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    text_fontsize = plot_args.get('text_fontsize', 14)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.55, 0.6))

    # handle plotting
    if axes:
        chained = True
    else:
        chained = False

    # supply fixed arguments to plots
    # might want to add other defaults here
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    ax = plot_histogram(evaluation_result.test_distribution,
                        evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with quantile values
    if not chained:
        try:
            ax.annotate('$\gamma = P(X \geq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile,
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)
        except TypeError:
            # if both quantiles are provided, we want to plot the greater-equal quantile
            ax.annotate('$\gamma = P(X \geq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile[0],
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if tight_layout:
        var = ax.get_figure().tight_layout
        ()

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_distribution_test(evaluation_result, axes=None, show=True,
                           plot_args=None):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the M-test.

    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead

    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure

    """
    warnings.warn(
        "'plot_distribution_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    plot_args = plot_args or {}
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.get('filename', None)
    xlabel = plot_args.get('xlabel', '')
    ylabel = plot_args.get('ylabel', '')
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution,
                        evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.3f}$\n$\omega$ = {:.3f}'
                    .format(evaluation_result.quantile,
                            evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.5, 0.3),
                    fontsize=14)

    title = plot_args.get('title', evaluation_result.name)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_likelihood_test(evaluation_result, axes=None, show=True,
                         plot_args=None):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation for the L-test.


    .. deprecated:: 0.7.0
       This function is deprecated and will be removed in version 1.0.
       Please use :func:`~csep.utils.plots.plot_test_distribution` instead.

    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult
        axes (matplotlib.Axes): axes object used to chain this plot
        show (bool): if true, call pyplot.show()
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * text_fontsize: (:class:`float`) - default: 14
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True
        * percentile (:class:`float`) Critial region to shade on histogram - default: 95
        * bins: (:class:`str`) - Set binning type. see matplotlib.hist for more info - default: 'auto'
        * xy: (:class:`list`/:class:`tuple`) - default: (0.55, 0.3)

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure
    """
    warnings.warn(
        "'plot_likelihood_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    plot_args = plot_args or {}
    title = plot_args.get('title', 'Pseudo-likelihood Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'Pseudo likelihood')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    text_fontsize = plot_args.get('text_fontsize', 14)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.55, 0.3))

    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    ax = plot_histogram(evaluation_result.test_distribution,
                        evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        try:
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile,
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)
        except TypeError:
            # if both quantiles are provided, we want to plot the greater-equal quantile
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile[1],
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if tight_layout:
        ax.figure.tight_layout()

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()
    return ax


def plot_spatial_test(evaluation_result, axes=None, plot_args=None, show=True):
    """
    Plot spatial test result from catalog based forecast

    .. deprecated:: 0.7.0
    This function is deprecated and will be removed in version 1.0.
    Please use :func:`~csep.utils.plots.plot_test_distribution` instead


    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult
        axes (matplotlib.Axes): axes object used to chain this plot
        show (bool): if true, call pyplot.show()
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * text_fontsize: (:class:`float`) - default: 14
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True
        * percentile (:class:`float`) Critial region to shade on histogram - default: 95
        * bins: (:class:`str`) - Set binning type. see matplotlib.hist for more info - default: 'auto'
        * xy: (:class:`list`/:class:`tuple`) - default: (0.2, 0.6)

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure
    """
    warnings.warn(
        "'plot_spatial_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    plot_args = plot_args or {}
    title = plot_args.get('title', 'Spatial Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'Normalized pseudo-likelihood')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    text_fontsize = plot_args.get('text_fontsize', 14)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.2, 0.6))

    # handle plotting
    if axes:
        chained = True
    else:
        chained = False

    # supply fixed arguments to plots
    # might want to add other defaults here
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)

    ax = plot_histogram(evaluation_result.test_distribution,
                        evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        try:
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile,
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)
        except TypeError:
            # if both quantiles are provided, we want to plot the greater-equal quantile
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile[1],
                                evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
                        fontsize=text_fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if tight_layout:
        ax.figure.tight_layout()

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax



def plot_poisson_consistency_test(eval_results, normalize=False,
                                  one_sided_lower=False, axes=None,
                                  plot_args=None, show=False):
    """ Plots results from CSEP1 tests following the CSEP1 convention.

    .. deprecated:: 0.7.0
    This function is deprecated and will be removed in version 1.0.
    Please use :func:`~csep.utils.plots.plot_consistency_test` instead

    Note: All of the evaluations should be from the same type of evaluation, otherwise the results will not be
          comparable on the same figure.

    Args:
        results (list): Contains the tests results :class:`csep.core.evaluations.EvaluationResult` (see note above)
        normalize (bool): select this if the forecast likelihood should be normalized by the observed likelihood. useful
                          for plotting simulation based simulation tests.
        one_sided_lower (bool): select this if the plot should be for a one sided test
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * color: (:class:`float`/:class:`None`) If None, sets it to red/green according to :func:`_get_marker_style` - default: 'black'
        * linewidth: (:class:`float`) - default: 1.5
        * capsize: (:class:`float`) - default: 4
        * hbars:  (:class:`bool`)  Flag to draw horizontal bars for each model - default: True
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True

    Returns:
        ax (:class:`matplotlib.pyplot.axes` object)
    """
    warnings.warn(
        "'plot_spatial_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_consistency_test.html",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        results = list(eval_results)
    except TypeError:
        results = [eval_results]
    results.reverse()
    # Parse plot arguments. More can be added here
    if plot_args is None:
        plot_args = {}
    figsize = plot_args.get('figsize', None)
    title = plot_args.get('title', results[0].name)
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', '')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    xticks_fontsize = plot_args.get('xticks_fontsize', None)
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    color = plot_args.get('color', 'black')
    linewidth = plot_args.get('linewidth', None)
    capsize = plot_args.get('capsize', 4)
    hbars = plot_args.get('hbars', True)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    plot_mean = plot_args.get('mean', False)

    if axes is None:
        fig, ax = pyplot.subplots(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    xlims = []
    for index, res in enumerate(results):
        # handle analytical distributions first, they are all in the form ['name', parameters].
        if res.test_distribution[0] == 'poisson':
            plow = scipy.stats.poisson.ppf((1 - percentile / 100.) / 2.,
                                           res.test_distribution[1])
            phigh = scipy.stats.poisson.ppf(1 - (1 - percentile / 100.) / 2.,
                                            res.test_distribution[1])
            mean = res.test_distribution[1]
            observed_statistic = res.observed_statistic
        # empirical distributions
        else:
            if normalize:
                test_distribution = numpy.array(
                    res.test_distribution) - res.observed_statistic
                observed_statistic = 0
            else:
                test_distribution = numpy.array(res.test_distribution)
                observed_statistic = res.observed_statistic
            # compute distribution depending on type of test
            if one_sided_lower:
                plow = numpy.percentile(test_distribution, 100 - percentile)
                phigh = numpy.percentile(test_distribution, 100)
            else:
                plow = numpy.percentile(test_distribution,
                                        (100 - percentile) / 2.)
                phigh = numpy.percentile(test_distribution,
                                         100 - (100 - percentile) / 2.)
            mean = numpy.mean(test_distribution)

        if not numpy.isinf(
                observed_statistic):  # Check if test result does not diverges
            percentile_lims = numpy.abs(numpy.array([[mean - plow,
                                                      phigh - mean]]).T)
            ax.plot(observed_statistic, index,
                    _get_marker_style(observed_statistic, (plow, phigh),
                                      one_sided_lower))
            ax.errorbar(mean, index, xerr=percentile_lims,
                        fmt='ko' * plot_mean, capsize=capsize,
                        linewidth=linewidth, ecolor=color)
            # determine the limits to use
            xlims.append((plow, phigh, observed_statistic))
            # we want to only extent the distribution where it falls outside of it in the acceptable tail
            if one_sided_lower:
                if observed_statistic >= plow and phigh < observed_statistic:
                    # draw dashed line to infinity
                    xt = numpy.linspace(phigh, 99999, 100)
                    yt = numpy.ones(100) * index
                    ax.plot(xt, yt, linestyle='--', linewidth=linewidth,
                            color=color)

        else:
            print('Observed statistic diverges for forecast %s, index %i.'
                  ' Check for zero-valued bins within the forecast' % (
                      res.sim_name, index))
            ax.barh(index, 99999, left=-10000, height=1, color=['red'],
                    alpha=0.5)

    try:
        ax.set_xlim(*_get_axis_limits(xlims))
    except ValueError:
        raise ValueError('All EvaluationResults have infinite '
                         'observed_statistics')
    ax.set_yticks(numpy.arange(len(results)))
    ax.set_yticklabels([res.sim_name for res in results],
                       fontsize=ylabel_fontsize)
    ax.set_ylim([-0.5, len(results) - 0.5])
    if hbars:
        yTickPos = ax.get_yticks()
        if len(yTickPos) >= 2:
            ax.barh(yTickPos, numpy.array([99999] * len(yTickPos)),
                    left=-10000,
                    height=(yTickPos[1] - yTickPos[0]), color=['w', 'gray'],
                    alpha=0.2, zorder=0)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.tick_params(axis='x', labelsize=xticks_fontsize)
    if tight_layout:
        ax.figure.tight_layout()
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax

def _get_marker_style(obs_stat, p, one_sided_lower):
    """Returns matplotlib marker style as fmt string"""
    if obs_stat < p[0] or obs_stat > p[1]:
        # red circle
        fmt = 'ro'
    else:
        # green square
        fmt = 'gs'
    if one_sided_lower:
        if obs_stat < p[0]:
            fmt = 'ro'
        else:
            fmt = 'gs'
    return fmt


def _get_axis_limits(pnts, border=0.05):
    """Returns a tuple of x_min and x_max given points on plot."""
    x_min = numpy.min(pnts)
    x_max = numpy.max(pnts)
    xd = (x_max - x_min) * border
    return (x_min - xd, x_max + xd)



@wraps(_plot_comparison_test)
def plot_comparison_test(*args, **kwargs):
    if 'axes' in kwargs:
        warnings.warn(
            "'plot_comparison_test' was called using the legacy 'axes' argument, which is deprecated.\n"
            "Please update your call to use:\n"
            "    plot_comparison_test(..., ax=<matplotlib axis>, ...)\n\n"
            "For documentation, see:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_comparison_test.html",
            DeprecationWarning,
            stacklevel=2
        )
        kwargs['ax'] = kwargs.pop('axes')

    return _plot_comparison_test(*args, **kwargs)

def _get_marker_t_color(distribution):
    """Returns matplotlib marker style as fmt string"""
    if distribution[0] > 0. and distribution[1] > 0.:
        fmt = 'green'
    elif distribution[0] < 0. and distribution[1] < 0.:
        fmt = 'red'
    else:
        fmt = 'grey'

    return fmt


def _get_marker_w_color(distribution, percentile):
    """Returns matplotlib marker style as fmt string"""

    if distribution < (1 - percentile / 100):
        fmt = True
    else:
        fmt = False

    return fmt


@wraps(_plot_consistency_test)
def plot_consistency_test(*args, **kwargs):
    """
    Legacy-compatible wrapper for plot_consistency_test.

    Detects deprecated usage (e.g., 'plot_args', 'axes') and falls back to legacy logic.
    """
    is_legacy_call = False

    # Detect legacy-style call (e.g., positional or 'plot_args' / 'axes' / 'variance')
    if len(args) > 1 or any(k in kwargs for k in ("plot_args", "axes", "variance")):
        is_legacy_call = True

    if is_legacy_call:
        warnings.warn(
            "'plot_consistency_test' was called using deprecated arguments and will fall back to legacy behavior.\n"
            "This fallback is deprecated and will be removed in version 1.0.\n"
            "Please update your usage to the new function signature:\n"
            "  plot_consistency_test(eval_results, normalize=False, one_sided_lower=False, percentile=95, ...)\n\n"
            "Refer to the updated documentation:\n"
            "  https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_consistency_test.html",
            DeprecationWarning,
            stacklevel=2
        )
        return _plot_consistency_test_legacy_impl(*args, **kwargs)

    return _plot_consistency_test(*args, **kwargs)

def _plot_consistency_test_legacy_impl(eval_results, normalize=False, axes=None,
                          one_sided_lower=False, variance=None, plot_args=None,
                          show=False):
    """ Plots results from CSEP1 tests following the CSEP1 convention.

    Note: All of the evaluations should be from the same type of evaluation, otherwise the results will not be
          comparable on the same figure.

    Args:
        eval_results (list): Contains the tests results :class:`csep.core.evaluations.EvaluationResult` (see note above)
        normalize (bool): select this if the forecast likelihood should be normalized by the observed likelihood. useful
                          for plotting simulation based simulation tests.
        one_sided_lower (bool): select this if the plot should be for a one sided test
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below

    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * color: (:class:`float`/:class:`None`) If None, sets it to red/green according to :func:`_get_marker_style` - default: 'black'
        * linewidth: (:class:`float`) - default: 1.5
        * capsize: (:class:`float`) - default: 4
        * hbars:  (:class:`bool`)  Flag to draw horizontal bars for each model - default: True
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True

    Returns:
        ax (:class:`matplotlib.pyplot.axes` object)
    """

    try:
        results = list(eval_results)
    except TypeError:
        results = [eval_results]
    results.reverse()
    # Parse plot arguments. More can be added here
    if plot_args is None:
        plot_args = {}
    figsize = plot_args.get('figsize', None)
    title = plot_args.get('title', results[0].name)
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', '')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    xticks_fontsize = plot_args.get('xticks_fontsize', None)
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    color = plot_args.get('color', 'black')
    linewidth = plot_args.get('linewidth', None)
    capsize = plot_args.get('capsize', 4)
    hbars = plot_args.get('hbars', True)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    plot_mean = plot_args.get('mean', False)

    if axes is None:
        fig, ax = pyplot.subplots(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    xlims = []

    for index, res in enumerate(results):
        # handle analytical distributions first, they are all in the form ['name', parameters].
        if res.test_distribution[0] == 'poisson':
            plow = scipy.stats.poisson.ppf((1 - percentile / 100.) / 2.,
                                           res.test_distribution[1])
            phigh = scipy.stats.poisson.ppf(1 - (1 - percentile / 100.) / 2.,
                                            res.test_distribution[1])
            mean = res.test_distribution[1]
            observed_statistic = res.observed_statistic

        elif res.test_distribution[0] == 'negative_binomial':
            var = variance
            observed_statistic = res.observed_statistic
            mean = res.test_distribution[1]
            upsilon = 1.0 - ((var - mean) / var)
            tau = (mean ** 2 / (var - mean))
            plow = scipy.stats.nbinom.ppf((1 - percentile / 100.) / 2., tau,
                                           upsilon)
            phigh = scipy.stats.nbinom.ppf(1 - (1 - percentile / 100.) / 2.,
                                          tau, upsilon)

        # empirical distributions
        else:
            if normalize:
                test_distribution = numpy.array(
                    res.test_distribution) - res.observed_statistic
                observed_statistic = 0
            else:
                test_distribution = numpy.array(res.test_distribution)
                observed_statistic = res.observed_statistic
            # compute distribution depending on type of test
            if one_sided_lower:
                plow = numpy.percentile(test_distribution, 5)
                phigh = numpy.percentile(test_distribution, 100)
            else:
                plow = numpy.percentile(test_distribution, 2.5)
                phigh = numpy.percentile(test_distribution, 97.5)
            mean = numpy.mean(res.test_distribution)

        if not numpy.isinf(
                observed_statistic):  # Check if test result does not diverges
            percentile_lims = numpy.array([[mean - plow, phigh - mean]]).T
            ax.plot(observed_statistic, index,
                    _get_marker_style(observed_statistic, (plow, phigh),
                                      one_sided_lower))
            ax.errorbar(mean, index, xerr=percentile_lims,
                        fmt='ko' * plot_mean, capsize=capsize,
                        linewidth=linewidth, ecolor=color)
            # determine the limits to use
            xlims.append((plow, phigh, observed_statistic))
            # we want to only extent the distribution where it falls outside of it in the acceptable tail
            if one_sided_lower:
                if observed_statistic >= plow and phigh < observed_statistic:
                    # draw dashed line to infinity
                    xt = numpy.linspace(phigh, 99999, 100)
                    yt = numpy.ones(100) * index
                    ax.plot(xt, yt, linestyle='--', linewidth=linewidth,
                            color=color)

        else:
            print('Observed statistic diverges for forecast %s, index %i.'
                  ' Check for zero-valued bins within the forecast' % (
                      res.sim_name, index))
            ax.barh(index, 99999, left=-10000, height=1, color=['red'],
                    alpha=0.5)

    try:
        ax.set_xlim(*_get_axis_limits(xlims))
    except ValueError:
        raise ValueError(
            'All EvaluationResults have infinite observed_statistics')
    ax.set_yticks(numpy.arange(len(results)))
    ax.set_yticklabels([res.sim_name for res in results],
                       fontsize=ylabel_fontsize)
    ax.set_ylim([-0.5, len(results) - 0.5])
    if hbars:
        yTickPos = ax.get_yticks()
        if len(yTickPos) >= 2:
            ax.barh(yTickPos, numpy.array([99999] * len(yTickPos)),
                    left=-10000,
                    height=(yTickPos[1] - yTickPos[0]), color=['w', 'gray'],
                    alpha=0.2, zorder=0)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.tick_params(axis='x', labelsize=xticks_fontsize)
    if tight_layout:
        ax.figure.tight_layout()
        fig.tight_layout()

    if show:
        pyplot.show()

    return ax


@wraps(_plot_concentration_ROC_diagram)
def plot_concentration_ROC_diagram(*args, **kwargs):
    """
    Backward-compatible wrapper for `plot_ROC_diagram`.

    Supports legacy arguments like 'axes' and 'plot_args'. Will be removed in version 1.0.
    """
    is_legacy_call = False

    if 'axes' in kwargs:
        is_legacy_call = True
        kwargs['ax'] = kwargs.pop('axes')

    if 'plot_args' in kwargs:
        is_legacy_call = True
        kwargs.update(kwargs.pop('plot_args'))

    if is_legacy_call:
        warnings.warn(
            "'plot_concentration_ROC_diagram' was called with legacy-style arguments (e.g. 'axes', 'plot_args').\n"
            "As of version 0.7.0, this usage is deprecated and will be removed in version 1.0.\n\n"
            "Please update your call to use keyword arguments directly, e.g.:\n"
            "    plot_ROC_diagram(..., ax=<axis>, ...)\n\n"
            "Documentation:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_ROC_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )

    return _plot_concentration_ROC_diagram(*args, **kwargs)


@wraps(_plot_ROC_diagram)
def plot_ROC_diagram(*args, **kwargs):
    """
    Backward-compatible wrapper for `plot_ROC_diagram`.

    Supports legacy arguments like 'axes' and 'plot_args'. Will be removed in version 1.0.
    """
    is_legacy_call = False

    if 'axes' in kwargs:
        is_legacy_call = True
        kwargs['ax'] = kwargs.pop('axes')

    if 'plot_args' in kwargs:
        is_legacy_call = True
        kwargs.update(kwargs.pop('plot_args'))

    if is_legacy_call:
        warnings.warn(
            "'plot_ROC_diagram' was called with legacy-style arguments (e.g. 'axes', 'plot_args').\n"
            "As of version 0.7.0, this usage is deprecated and will be removed in version 1.0.\n\n"
            "Please update your call to use keyword arguments directly, e.g.:\n"
            "    plot_ROC_diagram(..., ax=<axis>,...)\n\n"
            "Documentation:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_ROC_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )

    return _plot_ROC_diagram(*args, **kwargs)


@wraps(_plot_Molchan_diagram)
def plot_Molchan_diagram(*args, **kwargs):
    """
    Backward-compatible wrapper for `plot_Molchan_diagram`.

    Supports legacy arguments like 'axes' and 'plot_args'. Will be removed in version 1.0.
    """
    is_legacy_call = False

    if 'axes' in kwargs:
        is_legacy_call = True
        kwargs['ax'] = kwargs.pop('axes')

    if 'plot_args' in kwargs:
        is_legacy_call = True
        kwargs.update(kwargs.pop('plot_args'))

    if is_legacy_call:
        warnings.warn(
            "'plot_Molchan_diagram' was called with legacy-style arguments (e.g. 'axes', 'plot_args').\n"
            "As of version 0.7.0, this usage is deprecated and will be removed in version 1.0.\n\n"
            "Please update your call to use keyword arguments directly, e.g.:\n"
            "    plot_Molchan_diagram(..., ax=<axis>, ...)\n\n"
            "Documentation:\n"
            "    https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_Molchan_diagram.html",
            DeprecationWarning,
            stacklevel=2
        )

    return _plot_Molchan_diagram(*args, **kwargs)


def plot_pvalues_and_intervals(test_results, ax, var=None):
    """ Plots p-values and intervals for a list of Poisson or NBD test results

    Args:
        test_results (list): list of EvaluationResults for N-test. All tests should use the same distribution (ie Poisson or NBD).
        ax (matplotlib.axes.Axes.axis): axes to use for plot. create using matplotlib
        var (float): variance of the NBD distribution. Must be used for NBD plots.

    Returns:
        ax (matplotlib.axes.Axes.axis): axes handle containing this plot

    Raises:
        ValueError: throws error if NBD tests are supplied without a variance
    """
    warnings.warn(
        "'plot_pvalues_and_intervals' is being modified for version 1.0.\n"
        "Refer to the documentation:\n"
        "  https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_pvalues_and_intervals.html",
        DeprecationWarning,
        stacklevel=2
    )
    variance = var
    percentile = 97.5
    p_values = []

    # Differentiate between N-tests and other consistency tests
    if test_results[0].name == 'NBD N-Test' or test_results[
        0].name == 'Poisson N-Test':
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], marker='o', color='red', lw=0,
                                    label=r'p < 10e-5', markersize=10,
                                    markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='#FF7F50',
                                    lw=0, label=r'10e-5 $\leq$ p < 10e-4',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='gold', lw=0,
                                    label=r'10e-4 $\leq$ p < 10e-3',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='white', lw=0,
                                    label=r'10e-3 $\leq$ p < 0.0125',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='skyblue',
                                    lw=0, label=r'0.0125 $\leq$ p < 0.025',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='blue', lw=0,
                                    label=r'p $\geq$ 0.025', markersize=10,
                                    markeredgecolor='k')]
        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor='k')
        # Act on Negative binomial tests
        if test_results[0].name == 'NBD N-Test':
            if var is None:
                raise ValueError(
                    "var must not be None if N-tests use the NBD distribution.")

            for i in range(len(test_results)):
                mean = test_results[i].test_distribution[1]
                upsilon = 1.0 - ((variance - mean) / variance)
                tau = (mean ** 2 / (variance - mean))
                phigh97 = scipy.stats.nbinom.ppf(
                    (1 - percentile / 100.) / 2., tau, upsilon
                )
                plow97 = scipy.stats.nbinom.ppf(
                    1 - (1 - percentile / 100.) / 2., tau, upsilon
                )
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i,
                            xerr=numpy.array([[low97, high97]]).T, capsize=4,
                            color='slategray', alpha=1.0, zorder=0)
                p_values.append(test_results[i].quantile[
                                    1] * 2.0)  # Calculated p-values according to Meletti et al., (2021)

                if p_values[i] < 10e-5:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='red', markersize=8, zorder=2)
                if p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='#FF7F50', markersize=8, zorder=2)
                if p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='gold', markersize=8, zorder=2)
                if p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='white', markersize=8, zorder=2)
                if p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='skyblue', markersize=8, zorder=2)
                if p_values[i] >= 0.025:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='blue', markersize=8, zorder=2)
        # Act on Poisson N-test
        if test_results[0].name == 'Poisson N-Test':
            for i in range(len(test_results)):
                plow97 = scipy.stats.poisson.ppf((1 - percentile / 100.) / 2.,
                                                 test_results[
                                                     i].test_distribution[1])
                phigh97 = scipy.stats.poisson.ppf(
                    1 - (1 - percentile / 100.) / 2.,
                    test_results[i].test_distribution[1])
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i,
                            xerr=numpy.array([[low97, high97]]).T, capsize=4,
                            color='slategray', alpha=1.0, zorder=0)
                p_values.append(test_results[i].quantile[1] * 2.0)
                if p_values[i] < 10e-5:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='red', markersize=8, zorder=2)
                elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='#FF7F50', markersize=8, zorder=2)
                elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='gold', markersize=8, zorder=2)
                elif p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='white', markersize=8, zorder=2)
                elif p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='skyblue', markersize=8, zorder=2)
                elif p_values[i] >= 0.025:
                    ax.plot(test_results[i].observed_statistic,
                            (len(test_results) - 1) - i, marker='o',
                            color='blue', markersize=8, zorder=2)
    # Operate on all other consistency tests
    else:
        for i in range(len(test_results)):
            plow97 = numpy.percentile(test_results[i].test_distribution, 2.5)
            phigh97 = numpy.percentile(test_results[i].test_distribution, 97.5)
            low97 = test_results[i].observed_statistic - plow97
            high97 = phigh97 - test_results[i].observed_statistic
            ax.errorbar(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i,
                        xerr=numpy.array([[low97, high97]]).T, capsize=4,
                        color='slategray', alpha=1.0, zorder=0)
            p_values.append(test_results[i].quantile)

            if p_values[i] < 10e-5:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o', color='red',
                        markersize=8, zorder=2)
            elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o',
                        color='#FF7F50', markersize=8, zorder=2)
            elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o', color='gold',
                        markersize=8, zorder=2)
            elif p_values[i] >= 10e-3 and p_values[i] < 0.025:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o', color='white',
                        markersize=8, zorder=2)
            elif p_values[i] >= 0.025 and p_values[i] < 0.05:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o',
                        color='skyblue', markersize=8, zorder=2)
            elif p_values[i] >= 0.05:
                ax.plot(test_results[i].observed_statistic,
                        (len(test_results) - 1) - i, marker='o', color='blue',
                        markersize=8, zorder=2)

        legend_elements = [
            matplotlib.lines.Line2D([0], [0], marker='o', color='red', lw=0,
                                    label=r'p < 10e-5', markersize=10,
                                    markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='#FF7F50',
                                    lw=0, label=r'10e-5 $\leq$ p < 10e-4',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='gold', lw=0,
                                    label=r'10e-4 $\leq$ p < 10e-3',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='white', lw=0,
                                    label=r'10e-3 $\leq$ p < 0.025',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='skyblue',
                                    lw=0, label=r'0.025 $\leq$ p < 0.05',
                                    markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='blue', lw=0,
                                    label=r'p $\geq$ 0.05', markersize=10,
                                    markeredgecolor='k')]

        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor='k')

    return ax


def add_labels_for_publication(figure, style='bssa', labelsize=16):
    """ Adds publication labels too the outside of a figure.

    .. deprecated:: 0.7.0
    This function is deprecated and will be removed in version 1.0.

     """

    warnings.warn(
        "'plot_spatial_test' is deprecated and will be removed in version 1.0.\n"
        "Use 'plot_test_distribution' instead.\n"
        "Documentation: https://docs.cseptesting.org/reference/generated/csep.utils.plots.plot_test_distribution.html",
        DeprecationWarning,
        stacklevel=2
    )
    all_axes = figure.get_axes()
    ascii_iter = iter(string.ascii_lowercase)
    for ax in all_axes:
        # check for colorbar and ignore for annotations
        if ax.get_label() == 'Colorbar':
            continue
        annot = next(ascii_iter)
        if style == 'bssa':
            ax.annotate(f'({annot})', (0.025, 1.025), xycoords='axes fraction',
                        fontsize=labelsize)

    return