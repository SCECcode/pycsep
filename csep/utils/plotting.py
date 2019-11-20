from collections import defaultdict

import matplotlib
from matplotlib import cm
from matplotlib.collections import PatchCollection

import time
import numpy
import matplotlib.pyplot as pyplot
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from csep.utils.constants import SECONDS_PER_DAY, CSEP_MW_BINS
from csep.utils.calc import bin1d_vec
from csep.utils.time import datetime_to_utc_epoch

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

def plot_cumulative_events_versus_time_dev(xdata, ydata, obs_data, plot_args, show=False):
    """
    
    
    Args:
        xdata (ndarray): time bins for plotting shape (N,) 
        ydata (ndarray or list like): ydata for plotting; shape (N,5) in order 2.5%Per, 25%Per, 50%Per, 75%Per, 97.5%Per
        obs_data (ndarry): same shape as xdata
        plot_args: 
        show: 

    Returns:

    """
    figsize = plot_args.get('figsize', None)
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    legend_loc = plot_args.get('legend_loc', 'best')
    title = plot_args.get('title', 'Cumulative Event Counts')
    xlabel = plot_args.get('xlabel', 'Days')

    fig, ax = pyplot.subplots(figsize=figsize)
    try:
        fifth_per = ydata[0,:]
        first_quar = ydata[1,:]
        med_counts = ydata[2,:]
        second_quar = ydata[3,:]
        nine_fifth = ydata[4,:]
    except:
        raise TypeError("ydata must be a [N,5] ndarray.")
    # plotting

    ax.plot(xdata, obs_data, color='black', label=obs_label)
    ax.plot(xdata, med_counts, color='red', label=sim_label)
    ax.fill_between(xdata, fifth_per, nine_fifth, color='red', alpha=0.2, label='5%-95%')
    ax.fill_between(xdata, first_quar, second_quar, color='red', alpha=0.5, label='25%-75%')
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Event Count')
    ax.set_title(title)
    # pyplot.subplots_adjust(right=0.75)
    # annotate the plot with information from catalog
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

def plot_cumulative_events_versus_time(stochastic_event_sets, observation, show=False, plot_args={}):
    """
    Same as below but performs the statistics on numpy arrays without using pandas data frames.

    Args:
        stochastic_event_sets:
        observation:
        show:
        plot_args:

    Returns:
        ax: matplotlib.Axes
    """
    print('Plotting cumulative event counts.')
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
    time_bins, dt = numpy.linspace(numpy.min(extreme_times), numpy.max(extreme_times), 100, endpoint=True, retstep=True)
    n_bins = time_bins.shape[0]
    binned_counts = numpy.zeros((n_cat, n_bins))
    for i, ses in enumerate(stochastic_event_sets):
        n_events = ses.catalog.shape[0]
        ses_origin_time = ses.get_epoch_times()
        inds = bin1d_vec(ses_origin_time, time_bins)
        for j in range(n_events):
            binned_counts[i, inds[j]] += 1
        if (i+1) % 1500 == 0:
            t1 = time.time()
            print(f"Processed {i+1} catalogs in {t1-t0} seconds.")
    t1 = time.time()
    print(f'Collected binned counts in {t1-t0} seconds.')
    summed_counts = numpy.cumsum(binned_counts, axis=1)

    # compute summary statistics for plotting
    fifth_per = numpy.percentile(summed_counts, 5, axis=0)
    first_quar = numpy.percentile(summed_counts, 25, axis=0)
    med_counts = numpy.percentile(summed_counts, 50, axis=0)
    second_quar = numpy.percentile(summed_counts, 75, axis=0)
    nine_fifth = numpy.percentile(summed_counts, 95, axis=0)
    # compute median for comcat catalog
    obs_binned_counts = numpy.zeros(n_bins)
    inds = bin1d_vec(observation.get_epoch_times(), time_bins)
    for j in range(observation.event_count):
        obs_binned_counts[inds[j]] += 1
    obs_summed_counts = numpy.cumsum(obs_binned_counts)

    # update time_bins for plotting
    millis_to_hours = 60*60*1000*24
    time_bins = (time_bins - time_bins[0])/millis_to_hours
    time_bins = time_bins + (dt/millis_to_hours)
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
    ax.fill_between(time_bins, fifth_per, nine_fifth, color='red', alpha=0.2, label='5%-95%')
    ax.fill_between(time_bins, first_quar, second_quar, color='red', alpha=0.5, label='25%-75%')
    ax.legend(loc=legend_loc)
    ax.set_xlabel('Days since Mainshock')
    ax.set_ylabel('Cumulative Event Count')
    ax.set_title(title)
    pyplot.subplots_adjust(right=0.75)
    # annotate the plot with information from catalog
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

def plot_magnitude_versus_time(catalog, filename=None, show=False, plot_args={}, **kwargs):
    """
    Plots magnitude versus linear time for an earthquake catalog.

    Catalog class must implement get_magnitudes() and get_datetimes() in order for this function to work correctly.

    Args:
        catalog (:class:`~csep.core.catalogs.AbstractBaseCatalog`): catalog to visualize

    Returns:
        (tuple): fig and axes handle
    """
    # get values from plotting args
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    xlabel = plot_args.get('xlabel', 'X')
    ylabel = plot_args.get('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    legend_loc = plot_args.get('legend_loc', 'best')
    title = plot_args.get('title', '')
    marker_size = plot_args.get('marker_size', 10)
    color = plot_args.get('color', 'blue')
    c = plot_args.get('c', None)
    clabel = plot_args.get('clabel', None)

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
    if c is not None:
        h = ax.scatter(days_elapsed, magnitudes, marker='.', s=marker_size, c=c, cmap=cm.get_cmap('jet'), **kwargs)
        cbar = fig.colorbar(h)
        cbar.set_label(clabel)
    else:
        ax.scatter(days_elapsed, magnitudes, marker='.', s=marker_size, color=color, **kwargs)


    # do some labeling of the figure
    ax.set_title(title, fontsize=16, color='black')
    ax.set_xlabel('Days Elapsed')
    ax.set_ylabel('Magnitude')
    fig.tight_layout()

    # # annotate the plot with information from catalog
    # if catalog is not None:
    #     try:
    #         ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    #     except:
    #         pass

    # handle displaying of figures
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)

    if show:
        pyplot.show()

    return ax

def plot_histogram(simulated, observation, bins='fd', percentile=None,
                   show=False, axes=None, catalog=None, plot_args={}):
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
        catalog (csep.AbstractBaseCatalog): used for annotating the figures
        plot_args (dict): additional plotting commands. TODO: Documentation

    Returns:
        axis: matplolib axes handle
    """
    # Plotting

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
        ax.axvline(x=observation, color='black', linestyle='--', label=obs_label)

    else:
        # remove any nan values
        observation = observation[~numpy.isnan(observation)]
        ax.hist(observation, bins=bins, label=obs_label, edgecolor=None, linewidth=0)

    # remove any potential nans from arrays
    simulated = numpy.array(simulated)
    simulated = simulated[~numpy.isnan(simulated)]

    if color:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, label=sim_label, color=color, edgecolor=None, linewidth=0)
    else:
        n, bin_edges, patches = ax.hist(simulated, bins=bins, label=sim_label, edgecolor=None, linewidth=0)

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
        upper_xlim = upper_xlim + 2*d_bin

        lower_xlim = numpy.percentile(simulated, 0.25)
        lower_xlim = numpy.min([lower_xlim, numpy.min(observation)])
        lower_xlim = lower_xlim - 2*d_bin

        try:
            ax.set_xlim([lower_xlim, upper_xlim])
        except ValueError:
            print('Ignoring observation in axis scaling because inf or -inf')
            upper_xlim = numpy.percentile(simulated, 99.75)
            upper_xlim = upper_xlim + 2*d_bin

            lower_xlim = numpy.percentile(simulated, 0.25)
            lower_xlim = lower_xlim - 2*d_bin

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

def plot_ecdf(x, ecdf, xv=None, show=False, plot_args = {}):
    """
    Plots empirical cumulative distribution function.
    """
    # get values from plotting args
    sim_label = plot_args.get('sim_label', 'Simulated')
    obs_label = plot_args.get('obs_label', 'Observation')
    xlabel = plot_args.get('xlabel', 'X')
    ylabel = plot_args.get('ylabel', '$P(X \leq x)$')
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    legend_loc = plot_args.get('legend_loc', 'best')
    filename = plot_args.get('filename', None)

    # make figure
    fig, ax = pyplot.subplots()
    ax.plot(x, ecdf, label=sim_label)
    if xv:
        ax.axvline(x=xv, color='black', linestyle='--', label=obs_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    # if catalog is not None:
    #     ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)

    if show:
        pyplot.show()

    return ax

def plot_magnitude_histogram_dev(ses_data, obs, plot_args, show=False):
    bin_edges, obs_hist = obs.binned_magnitude_counts(retbins=True)
    n_obs = numpy.sum(obs_hist)
    event_counts = numpy.sum(ses_data, axis=1)
    # normalize all histograms by counts in each
    scale = n_obs / event_counts
    # use broadcasting
    ses_data = ses_data * scale.reshape(-1,1)
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
    pc = matplotlib.collections.PatchCollection(rectangles, facecolor='blue', alpha=0.3, edgecolor='blue')
    ax.add_collection(pc)
    # plot whiskers
    sim_label = plot_args.get('sim_label', 'Simulated Catalogs')
    obs_label = plot_args.get('obs_label', 'Observed Catalog')
    xlim = plot_args.get('xlim', None)
    title = plot_args.get('title', "UCERF3-ETAS Histogram")
    filename = plot_args.get('filename', None)

    ax.errorbar(bin_edges_plot, u3etas_median, yerr=[u3etas_emin, u3etas_emax], xerr=0.8 * dmw / 2, fmt=' ',
                    label=sim_label, color='blue', alpha=0.7)
    ax.plot(bin_edges_plot, obs_hist, '.k', markersize=10, label=obs_label)
    ax.legend(loc='upper right')
    ax.set_xlim(xlim)
    ax.set_xlabel('Mw')
    ax.set_ylabel('Count')
    ax.set_title(title)
    # ax.annotate(str(comcat), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    # pyplot.subplots_adjust(right=0.75)
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)
    if show:
        pyplot.show()
    return ax

def plot_magnitude_histogram(u3catalogs, comcat, show=True, plot_args={}):
    # get list of magnitudes list of ndarray
    u3etas_mws = list(map(lambda x: x.get_magnitudes(), u3catalogs))
    obs_mw = comcat.get_magnitudes()
    n_obs = comcat.get_number_of_events()

    # get histogram at arbitrary values
    mws = CSEP_MW_BINS
    dmw = mws[1] - mws[0]

    def get_hist(x, mws, normed=True):
        n_temp = len(x)
        if normed and n_temp != 0:
            temp_scale = n_obs / n_temp
            hist = numpy.histogram(x, bins=mws)[0] * temp_scale
        else:
            hist = numpy.histogram(x, bins=mws)[0]
        return hist

    # get hist values
    u3etas_hist = numpy.array(list(map(lambda x: get_hist(x, mws), u3etas_mws)))
    obs_hist, bin_edges = numpy.histogram(obs_mw, bins=mws)
    bin_edges_plot = (bin_edges[1:] + bin_edges[:-1]) / 2

    figsize = plot_args.get('figsize', None)
    fig = pyplot.figure(figsize=figsize)
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
    xlim = plot_args.get('xlim', None)
    title=plot_args.get('title', "UCERF3-ETAS Histogram")
    xycoords = plot_args.get('xycoords', (1.00, 0.40))
    filename = plot_args.get('filename', None)

    pyplot.errorbar(bin_edges_plot, u3etas_median, yerr=[u3etas_emin, u3etas_emax], xerr=0.8 * dmw / 2, fmt=' ',
                    label=sim_label, color='blue', alpha=0.7)
    pyplot.plot(bin_edges_plot, obs_hist, '.k', markersize=10, label='Comcat')
    pyplot.legend(loc='upper right')
    pyplot.xlim(xlim)
    pyplot.xlabel('Mw')
    pyplot.ylabel('Count')
    pyplot.title(title)
    # ax.annotate(str(comcat), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)
    pyplot.subplots_adjust(right=0.75)
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)
    if show:
        pyplot.show()

def plot_spatial_dataset(gridded, region, show=False, plot_args={}):
    """

    Args:
        gridded: 1d numpy array with vals according to region
        region: Region class
        plot_args: arguments to various matplotlib functions.

    Returns:

    """
    # get spatial information for plotting
    extent = region.get_bbox()
    # plot using cartopy
    figsize = plot_args.get('figsize', None)
    title = plot_args.get('title', 'Spatial Dataset')
    clim = plot_args.get('clim', None)
    clabel = plot_args.get('clabel', '')
    filename = plot_args.get('filename', None)

    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    lons, lats = numpy.meshgrid(region.xs, region.ys)
    im = ax.pcolormesh(lons, lats, gridded)
    ax.set_extent(extent)
    ax.coastlines(color='black', resolution='110m', linewidth=1)
    ax.add_feature(cartopy.feature.STATES)
    im.set_clim(clim)
    # colorbar options
    cbar = fig.colorbar(im, ax=ax)
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
    ax.set_title(title, y=1.06)
    # this is a cartopy.GeoAxes
    if filename is not None:
        fig.savefig(filename + '.pdf')
        fig.savefig(filename + '.png', dpi=300)

    if show:
        pyplot.show()
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
    filename = plot_args.get('filename', None)
    xlabel = plot_args.get('xlabel', 'Event Count in Catalog')
    ylabel = plot_args.get('ylabel', 'Number of Catalogs')
    xy = plot_args.get('xy', (0.5, 0.3))

    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        try:
            ax.annotate('$\delta_1 = P(X \geq x) = {:.2f}$\n$\delta_2 = P(X \leq x) = {:.2f}$\n$\omega = {:d}$'
                    .format(*evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=xy,
                    fontsize=14)
        except:
            ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                        .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                        xycoords='axes fraction',
                        xy=xy,
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

def plot_magnitude_test(evaluation_result, axes=None, show=True, plot_args={}):
    """
    Takes result from evaluation and generates a specific histogram plot to show the results of the statistical evaluation
    for the M-test.


    Args:
        evaluation_result: object that implements the interface of EvaluationResult

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
    filename = plot_args.get('filename', None)
    xy = plot_args.get('xy', (0.55, 0.6))
    fixed_plot_args = {'xlabel': 'D* Statistic',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=xy,
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Magnitude Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax

def plot_distribution_test(evaluation_result, axes=None, show=True, plot_args={}):
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
    filename = plot_args.get('filename', None)
    xlabel = plot_args.get('xlabel', '')
    ylabel = plot_args.get('ylabel', '')
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.3f}$\n$\omega$ = {:.3f}'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.5, 0.3),
                    fontsize=14)

    title = plot_args.get('title', evaluation_result.name)
    ax.set_title(title, fontsize=14)
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
    filename = plot_args.get('filename', None)
    fixed_plot_args = {'xlabel': 'Pseudo Likelihood',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)
    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.55, 0.3),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Pseudo Likelihood Test')
    ax.set_title(title, fontsize=14)

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


    Args:
        evaluation_result:

    Returns:

    """
    plot_args = plot_args or {}
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.get('filename', None)
    fixed_plot_args = {'xlabel': 'Normalized Pseudo Likelihood',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)

    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins='fd',
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.2, 0.6),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Spatial Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_probability_test(evaluation_result, axes=None, plot_args=None, show=True):
    """


    Args:
        evaluation_result:

    Returns:

    """
    plot_args = plot_args or {}
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.get('filename', None)
    fixed_plot_args = {'xlabel': 'Spatial Probability',
                       'ylabel': 'Number of Catalogs',
                       'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)

    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins='fd',
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.2, 0.6),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Probability Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax