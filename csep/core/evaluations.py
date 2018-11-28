import matplotlib.pyplot as pyplot

from csep.utils.plotting import plot_ecdf
from csep.utils.stats import ecdf
from csep.utils.math import func_inverse


# IDEA: Use decorators to provide common functionality to different types of evaluations. This would create an object that
# the decorated functions become members of. Similarly to the way that unittest behaves, but with decorators as opposed to
# class definitions.

def number_test(stochastic_event_set, observation, plot=False, interp_kind='nearest', fill_value='extrapolate', plot_args={}):
    """
    Perform an N-Test on a stochastic event set and observation.

    Args:
        stochastic_event_set (list of :class:`~csep.core.catalogs.BaseCatalog`)
        observation (:class:`~csep.core.catalogs.BaseCatalog`)
        plot (bool): visualize: yes or no

    Note:
        Catalogs must implement get_number_of_events() method for this function to work.

    Returns:
        (p_value, ax): axes is None if plot=False
    """
    # get number of events for observations and simulations
    sim_counts = []
    for catalog in stochastic_event_set:
        sim_counts.append(catalog.get_number_of_events())

    observation_count = observation.get_number_of_events()

    # get function and interpolator based on empirical cdf
    x, cdf = ecdf(sim_counts)

    # p-value represents P(X <= x)
    p_value = func_inverse(x, cdf, observation_count, kind=interp_kind, fill_value=fill_value)

    # handle plotting
    ax = None
    if plot:
        # supply fixed arguments to plots
        # might want to add other defaults here
        show = plot_args.pop('show', 'False')
        filename = plot_args.pop('filename', None)
        fixed_plot_args = {'xlabel': 'Event Count',
                           'ylabel': 'Cumulative Probability',
                           'obs_label': observation.name,
                           'sim_label': catalog.name}
        plot_args.update(fixed_plot_args)
        ax = plot_ecdf(x, cdf, observation_count, catalog=observation, plot_args=plot_args, filename=filename)

        # annotate the plot with information from catalog
        ax.annotate('$P(X \leq x) = {:.5f}$'.format(p_value), xycoords='axes fraction', xy=(0.6, 0.3), fontsize=14)
        ax.set_title("CSEP2 Number Test", fontsize=14)

        if show:
            pyplot.show()

    return (p_value, ax)
