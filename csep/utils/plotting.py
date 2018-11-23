import time
import numpy
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates

from csep.utils.constants import SECONDS_PER_DAY
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
"""

def plot_cumulative_events_versus_time(stochastic_event_set, observation, filename=None, show=False):
    """
    Plots cumulative number of events against time for both the observed catalog and a stochastic event set.
    Initially bins events by week and computes.

    Args:
        stochastic_event_set (iterable): iterable of :class:`~csep.core.catalogs.BaseCatalog` objects
        observation (:class:`~csep.core.catalogs.BaseCatalog`): single catalog, typically observation catalog
        filename (str): filename of file to save, if not None will save to that file
        show (bool): whether to making blocking call to display figure

    Returns:
        pyplot.Figure: fig
    """
    print('Plotting cumulative event counts.')
    fig, ax = pyplot.subplots()

    # date formatting
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    # get dataframe representation for all catalogs
    f = lambda x: x.get_dataframe()
    t0 = time.time()
    df = pandas.concat(list(map(f, stochastic_event_set)))
    t1 = time.time()
    print('Converted {} ruptures from {} catalogs into a DataFrame in {} seconds.\n'
          .format(len(df), len(stochastic_event_set), t1-t0))

    # get counts, cumulative_counts, percentiles in weekly intervals
    df_comcat = observation.get_dataframe()

    # get statistics from stochastic event set
    # IDEA: make this a function, might want to re-use this binning
    df1 = df.groupby([df['catalog_id'], pandas.Grouper(freq='W')])['counts'].agg(['sum'])
    df1['cum_sum'] = df1.groupby(level=0).cumsum()
    df2 = df1.groupby('datetime').describe(percentiles=(0.05,0.5,0.95))

    # remove tz information so pandas can plot
    df2.index = df2.index.tz_localize(None)

    # get statistics from catalog
    df1_comcat = df_comcat.groupby(pandas.Grouper(freq='W'))['counts'].agg(['sum'])
    df1_comcat['obs_cum_sum'] = df1_comcat['sum'].cumsum()
    df1_comcat.index = df1_comcat.index.tz_localize(None)

    df2.columns = ["_".join(x) for x in df2.columns.ravel()]
    df3 = df2.merge(df1_comcat, left_index=True, right_on='datetime', left_on='datetime')

    # plotting
    ax.plot(df3.index, df3['obs_cum_sum'], color='black', label=observation.name + ' (Obs)')
    ax.plot(df3.index, df3['cum_sum_50%'], color='blue', label=stochastic_event_set[0].name)
    ax.fill_between(df3.index, df3['cum_sum_5%'], df3['cum_sum_95%'], color='blue', alpha=0.2, label='5%-95%')
    ax.legend(loc='lower right')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(df3.index.year.max())
    ax.set_ylabel('Cumulative Event Count')

    # save figure
    if filename is not None:
        fig.savefig(filename)

    # optionally show figure
    if show:
        pyplot.show()

    return ax

def plot_magnitude_versus_time(catalog, filename=None, show=False):
    """
    Plots magnitude versus linear time for an earthquake catalog.

    Catalog class must implement get_magnitudes() and get_datetimes() in order for this function to work correctly.

    Args:
        catalog (:class:`~csep.core.catalogs.BaseCatalog`): catalog to visualize

    Returns:
        (tuple): fig and axes handle
    """
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
    ax.scatter(days_elapsed, magnitudes, marker='.', s=10)

    # do some labeling of the figure
    ax.set_title(catalog.name, fontsize=16, color='black')
    ax.set_xlabel('Days Elapsed')
    ax.set_ylabel('Magnitude')
    fig.tight_layout()

    # handle displaying of figures
    if filename is not None:
        fig.savefig(filename)

    if show:
        pyplot.show()

    return ax

def plot_histogram(simulated, observation, bins='fd', filename=None, show=False, axes=None, catalog=None, plot_args = {}):
    """
    Plots histogram of single statistic for stochastic event sets and observations. The function will behave differently
    depending on the inputs.

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
    chained = True
    if axes is not None:
        ax = axes
    else:
        fig, ax = pyplot.subplots()
        chained = False

    # parse plotting arguments
    sim_label = plot_args.pop('sim_label', 'Simulated')
    obs_label = plot_args.pop('obs_label', 'Observation')
    xlabel = plot_args.pop('xlabel', 'X')
    ylabel = plot_args.pop('ylabel', 'Frequency')
    xycoords = plot_args.pop('xycoords', (1.00, 0.40))
    title = plot_args.pop('title', None)
    legend_loc = plot_args.pop('legend_loc', 'best')

    # this could throw an error exposing bad implementation
    observation = numpy.array(observation)

    if not chained:
        try:
            n = len(observation)
            ax.hist(observation, bins=bins, edgecolor='black', alpha=0.5, label=obs_label)
        except TypeError:
            ax.axvline(x=observation, color='black', linestyle='--', label=obs_label)

    simulated = numpy.array(simulated)
    ax.hist(simulated, bins=bins, edgecolor='black', alpha=0.5, label=sim_label)

    # annotate the plot with information from catalog
    if catalog is not None:
        ax.annotate(str(catalog), xycoords='axes fraction', xy=xycoords, fontsize=10, annotation_clip=False)

    # ax.set_xlim(left=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

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
        print('Computing MFD for catalog.')
        catalog.get_mfd()

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

def plot_ecdf(x, ecdf, xv, catalog=None, filename=None, show=False, plot_args = {}):
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
