import matplotlib
from matplotlib import pyplot as pyplot
from matplotlib.collections import PatchCollection

import time
import numpy
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from csep.utils.constants import SECONDS_PER_DAY, CSEP_MW_BINS
from csep.utils.time import epoch_time_to_utc_datetime

"""
This module contains plotting routines that generate figures for the stochastic event sets produced from
CSEP2 experiments.

Right now functions dont have consistent signatures. That means that some functions might have more functionality than others
while the routines are being developed.

TODO: Add annotations for other two plots.
TODO: Add ability to plot annotations from multiple catalogs. Esp, for plot_histogram()
IDEA: Same concept mentioned in evaluations might apply here. The plots could be a common class that might provide
      more control to the end user.
IDEA: Since plotting functions are usable by these classes only that don't implement iter routines, maybe make them a class
      method. like catalog.plot_thing()
"""


# TODO: Fix this to work with shorter interval catalogs, grouping defaults to week. should default to nchunks.
def plot_cumulative_events_versus_time(stochastic_event_set, observation, show=False, plot_args={}):
    """
    Plots cumulative number of events against time for both the observed catalog and a stochastic event set.
    Initially bins events by week and computes.

    Args:
        stochastic_event_set (iterable): iterable of :class:`~csep.core.catalogs.BaseCatalog` objects
        observation (:class:`~csep.core.catalogs.BaseCatalog`): single catalog, typically observation catalog
        filename (str): filename of file to save, if not None will save to that file
        show (bool): whether to making blocking call to display figure
        plot_args (dict): args pass onto the plot function from matplotlib.

    Returns:
        pyplot.Figure: fig
    """
    print('Plotting cumulative event counts.')
    fig, ax = pyplot.subplots(figsize=(12,9))

    # date formatting
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    # get dataframe representation for all catalogs
    f = lambda x: x.get_dataframe()
    t0 = time.time()
    cats = list(map(f, stochastic_event_set))
    df = pandas.concat(cats)
    t1 = time.time()
    print('Converted {} ruptures from {} catalogs into a DataFrame in {} seconds.\n'
          .format(len(df), len(cats), t1-t0))

    # get counts, cumulative_counts, percentiles in weekly intervals
    df_obs = observation.get_dataframe()

    # get statistics from stochastic event set
    # IDEA: make this a function, might want to re-use this binning
    df1 = df.groupby([df['catalog_id'], pandas.Grouper(freq='W')])['counts'].agg(['sum'])
    df1['cum_sum'] = df1.groupby(level=0).cumsum()
    df2 = df1.groupby('datetime').describe(percentiles=(0.05,0.25,0.5,0.75,0.95))

    # remove tz information so pandas can plot
    df2.index = df2.index.tz_localize(None)

    # get statistics from catalog
    df1_comcat = df_obs.groupby(pandas.Grouper(freq='W'))['counts'].agg(['sum'])
    df1_comcat['obs_cum_sum'] = df1_comcat['sum'].cumsum()
    df1_comcat.index = df1_comcat.index.tz_localize(None)

    df2.columns = ["_".join(x) for x in df2.columns.ravel()]
    df3 = df2.merge(df1_comcat, left_index=True, right_on='datetime', left_on='datetime')

    # get values from plotting args
    sim_label = plot_args.pop('sim_label', 'Simulated')
    obs_label = plot_args.pop('obs_label', 'Observation')
    xlabel = plot_args.pop('xlabel', 'X')
    ylabel = plot_args.pop('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.pop('xycoords', (1.00, 0.40))
    legend_loc = plot_args.pop('legend_loc', 'best')

    # plotting
    ax.plot(df3.index, df3['obs_cum_sum'], color='black', label=obs_label)
    ax.plot(df3.index, df3['cum_sum_50%'], color='blue', label=sim_label)
    ax.fill_between(df3.index, df3['cum_sum_5%'], df3['cum_sum_95%'], color='blue', alpha=0.2, label='5%-95%')
    ax.fill_between(df3.index, df3['cum_sum_25%'], df3['cum_sum_75%'], color='blue', alpha=0.5, label='25%-75%')
    ax.legend(loc=legend_loc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(df3.index.year.max())
    ax.set_ylabel('Cumulative Event Count')

    pyplot.subplots_adjust(right=0.75)

    # annotate the plot with information from catalog
    ax.annotate(str(observation), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    # save figure
    filename = plot_args.get('filename', None)
    if filename is not None:
        fig.savefig(filename)

    # optionally show figure
    if show:
        pyplot.show()

    return ax


def plot_magnitude_versus_time(catalog, filename=None, show=False, plot_args={}, **kwargs):
    """
    Plots magnitude versus linear time for an earthquake catalog.

    Catalog class must implement get_magnitudes() and get_datetimes() in order for this function to work correctly.

    Args:
        catalog (:class:`~csep.core.catalogs.BaseCatalog`): catalog to visualize

    Returns:
        (tuple): fig and axes handle
    """
    # get values from plotting args
    sim_label = plot_args.pop('sim_label', 'Simulated')
    obs_label = plot_args.pop('obs_label', 'Observation')
    xlabel = plot_args.pop('xlabel', 'X')
    ylabel = plot_args.pop('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.pop('xycoords', (1.00, 0.40))
    legend_loc = plot_args.pop('legend_loc', 'best')
    title = plot_args.pop('title', '')
    marker_size = plot_args.pop('marker_size', 10)
    color = plot_args.pop('color', 'blue')

    print('Plotting magnitude versus time.')
    fig = pyplot.figure(figsize=(8,3))
    ax = fig.add_subplot(111)

    # get time in days
    # plotting timestamps for now, until I can format dates on axis properly
    f = lambda x: numpy.array(x.timestamp()) / SECONDS_PER_DAY

    # map returns a generator function which we collapse with list
    days_elapsed = numpy.array(list(map(f, catalog.get_datetimes())))
    days_elapsed = days_elapsed - days_elapsed[0]

    magnitudes = catalog.get_magnitudes()

    # make plot
    ax.scatter(days_elapsed, magnitudes, marker='.', s=marker_size, color=color)

    # do some labeling of the figure
    ax.set_title(title, fontsize=16, color='black')
    ax.set_xlabel('Days Elapsed')
    ax.set_ylabel('Magnitude')
    fig.tight_layout()

    # annotate the plot with information from catalog
    if catalog is not None:
        ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    # handle displaying of figures
    if filename is not None:
        fig.savefig(filename)

    if show:
        pyplot.show()

    return ax


def plot_histogram(simulated, observation, bins='fd', percentile=None,
                   filename=None, show=False, axes=None, catalog=None, plot_args={}):
    """
    Plots histogram of single statistic for stochastic event sets and observations. The function will behave differently
    depending on the inumpyuts.

    Simulated should always be either a list or numpy.array where there would be one value per catalog in the stochastic event
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
        catalog (csep.BaseCatalog): used for annotating the figures
        plot_args (dict): additional plotting commands. TODO: Documentation

    Returns:
        axis: matplolib axes handle
    """
    # Plotting

    chained = False
    if axes is not None:
        chained = True
        ax = axes
    else:
        if catalog:
            fig, ax = pyplot.subplots(figsize=(12,9))
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

    # this could throw an error exposing bad implementation
    observation = numpy.array(observation)

    try:
        n = len(observation)
    except TypeError:
        ax.axvline(x=observation, color='black', linestyle='--', label=obs_label)
    else:
        # remove any nan values
        observation = observation[~numpy.isnan(observation)]
        ax.hist(observation, bins=bins, edgecolor='black', alpha=0.5, rwidth=0.75, label=obs_label)

    # remove any potential nans from arrays
    simulated = numpy.array(simulated)
    simulated = simulated[~numpy.isnan(simulated)]
    if color:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, edgecolor='black', rwidth=0.75, alpha=0.5, label=sim_label,
                                        color=color)
    else:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, edgecolor='black', rwidth=0.75, alpha=0.5, label=sim_label)

    # color bars for rejection area
    if percentile is not None:
        inc = (100 - percentile) / 2
        inc_high = 100 - inc
        inc_low = inc
        p_high = numpy.percentile(simulated, inc_high)
        idx_high = numpy.digitize(p_high, bin_edges)
        p_low = numpy.percentile(simulated, inc_low)
        idx_low = numpy.digitize(p_low, bin_edges)

    # annotate the plot with information from catalog
    if catalog is not None and not chained:
        ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
        pyplot.subplots_adjust(right=0.75)

    # show 99.9% of data
    upper_xlim = numpy.percentile(simulated, 99.95)
    upper_xlim = numpy.max([upper_xlim, numpy.max(observation)])
    d_bin = bin_edges[1] - bin_edges[0]
    upper_xlim = upper_xlim + 2*d_bin

    lower_xlim = numpy.percentile(simulated, 0.05)
    lower_xlim = numpy.min([0, lower_xlim, numpy.min(observation)])
    lower_xlim = lower_xlim - 2*d_bin

    ax.set_xlim([lower_xlim, upper_xlim])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend_loc)

    # hacky workaround for coloring legend, by calling after legend.
    if percentile is not None:
        for idx in range(idx_low):
            patches[idx].set_fc('red')

        for idx in range(idx_high, len(patches)):
            patches[idx].set_fc('red')

        if filename is not None:
            pyplot.savefig(filename)

    if show:
        pyplot.show()

    return ax


def plot_mfd(catalog, filename=None, show=False, **kwargs):
    """
    Plots MFD from CSEP Catalog.

    Example usage would be:
    >>> plot_mfd(catalog, show=True)

    Args:
        catalog (:class:`~csep.core.catalogs.BaseCatalog`): instance of catalog class
        filename (str): filename to save figure
        show (bool): render figure locally using matplotlib backend

    Returns:
        ax (Axis): matplotlib axis handle
    """
    fig, ax = pyplot.subplots()

    mfd = catalog.mfd
    if mfd is None:
        print('Computing MFD for catalog {}.'.format(catalog.name))
        mfd = catalog.get_mfd()

    # get other vals for plotting
    a = mfd.loc[0,'a']
    b = mfd.loc[0,'b']
    ci_b = mfd.loc[0,'ci_b']

    # take mid point of magnitude bins for plotting
    x = numpy.array(mfd.index.categories.mid)
    try:
        ax.scatter(x, mfd['counts'], color='black', label='{} (accessed: {})'
                      .format(catalog.name, catalog.date_accessed.date()))
    except:
        ax.scatter(x, mfd['counts'], color='black', label='{}'.format(catalog.name))

    plt_label = '$log(N)={}-{}\pm{}M$'.format(numpy.round(a,2),numpy.round(abs(b),2),numpy.round(numpy.abs(ci_b),2))
    ax.plot(x, 10**mfd['N_est'], label=plt_label)
    ax.fill_between(x, 10**mfd['lower_ci'], 10**mfd['upper_ci'], color='blue', alpha=0.2)

    # annotations
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Magnitude Frequency Distribution')
    ax.annotate(s='Start Date: {}\nEnd Date: {}\n\nLatitude: ({:.2f}, {:.2f})\nLongitude: ({:.2f}, {:.2f})'
                .format(catalog.start_time.date(), catalog.end_time.date(),
                       catalog.min_latitude,catalog.max_latitude,
                       catalog.min_longitude,catalog.max_longitude),
                xycoords='axes fraction', xy=(0.5, 0.65), fontsize=10)
    ax.legend(loc='lower left')

    # handle saving
    if filename:
        pyplot.savefig(filename)
    if show:
        pyplot.show()


def plot_ecdf(x, ecdf, xv=None, catalog=None, filename=None, show=False, plot_args = {}):
    """
    Plots empirical cumulative distribution function.
    """
    # get values from plotting args
    sim_label = plot_args.pop('sim_label', 'Simulated')
    obs_label = plot_args.pop('obs_label', 'Observation')
    xlabel = plot_args.pop('xlabel', 'X')
    ylabel = plot_args.pop('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.pop('xycoords', (1.00, 0.40))
    legend_loc = plot_args.pop('legend_loc', 'best')

    # make figure
    fig, ax = pyplot.subplots()
    ax.plot(x, ecdf, label=sim_label)
    if xv:
        ax.axvline(x=xv, color='black', linestyle='--', label=obs_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    if catalog is not None:
        ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    if filename is not None:
        pyplot.savefig(filename)

    if show:
        pyplot.show()

    return ax


def plot_magnitude_histogram(u3catalogs, comcat, show=True, plot_args={}):
    # get list of magnitudes list of ndarray
    u3etas_mws = list(map(lambda x: x.get_magnitudes(), u3catalogs))
    obs_mw = comcat.get_magnitudes()
    n_obs = comcat.get_number_of_events()

    # get ecdf at arbitrary values
    mws = CSEP_MW_BINS
    dmw = mws[1] - mws[0]


    def get_hist(x, mws, normed=True):
        n_temp = len(x)
        temp_scale = n_obs / n_temp
        if normed:
            hist = numpy.histogram(x, bins=mws)[0] * temp_scale
        else:
            hist = numpy.histogram(x, bins=mws)[0]
        return hist

    # get hist values
    u3etas_hist = numpy.array(list(map(lambda x: get_hist(x, mws), u3etas_mws)))
    obs_hist, bin_edges = numpy.histogram(obs_mw, bins=mws)
    bin_edges_plot = (bin_edges[1:] + bin_edges[:-1]) / 2

    fig = pyplot.figure(figsize=(12,9))
    ax = fig.gca()
    u3etas_median = numpy.median(u3etas_hist, axis=0)
    u3etas_low = numpy.percentile(u3etas_hist, 2.5, axis=0)
    u3etas_high = numpy.percentile(u3etas_hist, 97.5, axis=0)
    u3etas_min = numpy.min(u3etas_hist, axis=0)
    u3etas_max = numpy.max(u3etas_hist, axis=0)
    u3etas_emax = u3etas_max - u3etas_median
    u3etas_emin = u3etas_median - u3etas_min

    # u3etas_emax = u3etas_max
    # plot 95% range as rectangles
    rectangles = []
    for i in range(len(mws) - 1):
        width = dmw / 2
        height = u3etas_high[i] - u3etas_low[i]
        xi = mws[i] + width / 2
        yi = u3etas_low[i]
        rect = matplotlib.patches.Rectangle((xi, yi), width, height)
        rectangles.append(rect)
    pc = matplotlib.collections.PatchCollection(rectangles, facecolor='blue', alpha=0.3, edgecolor='blue')
    ax.add_collection(pc)
    # plot whiskers
    sim_label = plot_args.get('sim_label', 'Simulated Catalogs')
    pyplot.errorbar(bin_edges_plot, u3etas_median, yerr=[u3etas_emin, u3etas_emax], xerr=0.8 * dmw / 2, fmt=' ',
                    label=sim_label, color='blue', alpha=0.7)
    pyplot.plot(bin_edges_plot, obs_hist, '.k', markersize=10, label='Comcat')
    pyplot.legend(loc='upper right')
    xlim = plot_args.get('xlim', None)
    pyplot.xlim(xlim)
    pyplot.xlabel('Mw')
    pyplot.ylabel('Count')
    pyplot.title("UCERF3-ETAS Histogram")
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    ax.annotate(str(comcat), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    pyplot.subplots_adjust(right=0.75)
    if show:
        pyplot.show()


def plot_spatial_dataset(gridded, region, plot_args={}):
    """
    plots spatial dataset associated with region

    Args:
        gridded: 1d numpy array with vals according to region
        region: Region class
        plot_args: arguments to various matplotlib functions.

    Returns:

    """
    # get spatial information for plotting
    extent = region.get_bbox()
    # plot using cartopy
    figsize = plot_args.get('figsize', (16,9))
    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    lons, lats = numpy.meshgrid(region.xs, region.ys)
    im = ax.pcolormesh(lons, lats, gridded)
    ax.set_extent(extent)
    ax.coastlines(color='black', resolution='110m', linewidth=1)
    ax.add_feature(cartopy.feature.STATES)
    clim = plot_args.get('clim', None)
    im.set_clim(clim)
    # colorbar options
    cbar = fig.colorbar(im, ax=ax)
    clabel = plot_args.get('clabel', '')
    cbar.set_label(clabel)
    # matplotlib.cm.get_cmap().set_bad(color='white')
    # matplotlib.cm.get_cmap().set_under(color='gray')
    # gridlines options
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.xlines = False
    gl.ylines = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # this is a cartopy.GeoAxes
    return ax


def plot_number_test(evaluation_result, axes=None, show=True, plot_args={}):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the n-test.


    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure

    """
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.pop('filename', None)
    fixed_plot_args = {'xlabel': 'Event Counts per Catalog',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.pop('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\delta_1 = P(X \geq x) = {:.5f}$\n$\delta_2 = P(X \leq x) = {:.5f}$'
                    .format(*evaluation_result.quantile),
                    xycoords='axes fraction',
                    xy=(0.5, 0.3),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Number Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        pyplot.savefig(filename)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax

def plot_magnitude_test(evaluation_result, axes=None, show=True, plot_args={}):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the M-test.


    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure

    """
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.pop('filename', None)
    fixed_plot_args = {'xlabel': 'D* Statistic',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.pop('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.5f}$'
                    .format(evaluation_result.quantile),
                    xycoords='axes fraction',
                    xy=(0.5, 0.3),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Magnitude Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        pyplot.savefig(filename)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_likelihood_test(evaluation_result, axes=None, show=True, plot_args={}):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the L-test.


    Args:
        evaluation_result: object-like var that implements the interface of the above EvaluationResult

    Returns:
        ax (matplotlib.axes.Axes): can be used to modify the figure

    """
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.pop('filename', None)
    fixed_plot_args = {'xlabel': 'Pseudo Likelihood',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.pop('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.5f}$'
                    .format(evaluation_result.quantile),
                    xycoords='axes fraction',
                    xy=(0.5, 0.3),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Pseudo Likelihood Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        pyplot.savefig(filename)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_spatial_test(evaluation_result, axes=None, plot_args={}, show=True):
    """


    Args:
        evaluation_result:

    Returns:

    """
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.pop('filename', None)
    fixed_plot_args = {'xlabel': 'Normalized Pseudo Likelihood',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.pop('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.5f}$'
                    .format(evaluation_result.quantile),
                    xycoords='axes fraction',
                    xy=(0.2, 0.7),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Spatial Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        pyplot.savefig(filename)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax