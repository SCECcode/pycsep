import matplotlib.pyplot as pyplot
import numpy
from csep.utils.constants import SECONDS_PER_DAY

"""
This module contains plotting routines that generate figures for the stochastic event sets produced from
CSEP2 experiments.

Example:
    TODO: Write example for the plotting functions.
    TODO: Ensure this function works with generators using itertools.islice
"""


def plot_cumulative_events_versus_time(stochastic_event_set, observation, fig=None,
                                            filename=None, show=False):
    """
    Plots cumulative number of events against time for both the observed catalog and a stochastic event set.
    Plots the catalog with the median number of events.

    Args:
        stochastic_event_set (iterable): iterable of :class:`~csep.core.catalogs.BaseCatalog` objects
        observation (:class:`~csep.core.catalogs.BaseCatalog`): single catalog, typically observation catalog
        filename (str): filename of file to save, if not None will save to that file
        show (bool): whether to making blocking call to display figure

    Returns:
        pyplot.Figure: fig
    """
    # set up figure
    chained = False
    if fig is not None:
        chained = True

    fig = fig or pyplot.figure()
    ax = fig.gca()

    # need to get the index of the stochastc event sets to plot
    ses_event_counts = []
    for catalog in stochastic_event_set:
        ses_event_counts.append(catalog.get_number_of_events())
    ses_event_counts = numpy.array(ses_event_counts)

    # get index of median
    ind_med = abs(ses_event_counts-numpy.percentile(ses_event_counts, 50, interpolation='nearest')).argmin()

    # plot median index
    cat_med = stochastic_event_set[ind_med]

    # plotting timestamps for now, until I can format dates on axis properly
    f = lambda x: numpy.array(x.timestamp()) / SECONDS_PER_DAY

    days_med = numpy.array(list(map(f, cat_med.get_datetimes())))
    days_med_zero = days_med - days_med[0]

    days_obs = numpy.array(list(map(f, observation.get_datetimes())))
    days_obs_zero = days_obs - days_obs[0]

    ax.plot(days_med_zero, cat_med.get_cumulative_number_of_events(), label=cat_med.name)

    if not chained:
        ax.plot(days_obs_zero, observation.get_cumulative_number_of_events(), '-k', label=observation.name)

    # do some labeling
    ax.set_title(cat_med.name, fontsize=16, color='black')
    ax.set_xlabel('Days Elapsed')
    ax.set_ylabel('Cumulative Number of Events')
    ax.legend(loc='best')

    # save figure
    if filename is not None:
        fig.savefig(filename)

    if show:
        pyplot.show()

    return fig

def plot_magnitude_versus_time(catalog, filename=None, show=False):
    """
    Plots magnitude versus linear time for an earthquake catalog.

    Catalog class must implement get_magnitudes() and get_datetimes() in order for this function to work correctly.

    Args:
        catalog (:class:`~csep.core.catalogs.BaseCatalog`): catalog to visualize

    Returns:
        (tuple): fig and axes handle
    """
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

    return fig

def plot_histogram(simulated, observation, filename=None, show=False, **kwargs):
    """
    Plots histogram of simulated values and shows observation with vertical line.

    Args:
        simulated (numpy.array): numpy.array like representation of statistics computed from catalogs.

    Returns:
        (tuple): fig and axes handle
    """
    raise NotImplementedError('plot_histogram has not been implemented.')
