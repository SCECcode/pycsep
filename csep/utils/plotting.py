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

Example:
    TODO: Write example for the plotting functions.
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

def plot_histogram(simulated, observation, filename=None, show=False, **kwargs):
    """
    Plots histogram of simulated values and shows observation with vertical line.

    Args:
        simulated (numpy.array): numpy.array like representation of statistics computed from catalogs.

    Returns:
        (tuple): fig and axes handle
    """
    raise NotImplementedError('plot_histogram has not been implemented.')

def plot_mfd(catalog, filename=None, show=False, **kwargs):
    """
    Plots MFD from pandas DataFrame.
    In theory could plot from anything that is dict-like with plottable arrays in the correct fields.

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

    # get other vals for plotting
    a = mfd['a'].iloc[0]
    b = mfd['b'].iloc[0]
    ci_b = mfd['ci_b'].iloc[0]

    # take mid point of magnitude bins for plotting
    idx = numpy.array(mfd.index.categories.mid)
    try:
        ax.scatter(idx, mfd['counts'], color='black', label='{} (accessed: {})'
                      .format(catalog.name, catalog.date_accessed.date()))
    except:
        ax.scatter(idx, mfd['counts'], color='black', label='{}'.format(catalog.name))
    ax.plot(idx, 10**mfd['N_est'], label='$log(N)={}-{}\pm{}M$'.format(numpy.round(a,2),numpy.round(abs(b),2),numpy.round(numpy.abs(ci_b),2)))
    ax.fill_between(idx, 10**mfd['lower_ci'], 10**mfd['upper_ci'], color='blue', alpha=0.2)

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
