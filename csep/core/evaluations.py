import matplotlib.pyplot as pyplot

from csep.utils.plotting import plot_ecdf
from csep.utils.stats import less_equal_ecdf, greater_equal_ecdf, ecdf


# we might want to consider removing the return axis functionality from this function.
def number_test(stochastic_event_set, observation, plot=False, show=False, plot_args={}):
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
        sim_counts.append(catalog.event_count)
    observation_count = observation.event_count

    # get delta_1 and delta_2 values
    delta_1, delta_2 = _number_test(sim_counts, observation_count)

    # handle plotting
    if plot:
        ax = None
        # supply fixed arguments to plots
        # might want to add other defaults here
        show = plot_args.pop('show', 'False')
        filename = plot_args.pop('filename', None)
        fixed_plot_args = {'xlabel': 'Event Count',
                           'ylabel': 'Cumulative Probability',
                           'obs_label': observation.name,
                           # explicitly assumes that all catalogs in list are of the same type
                           'sim_label': stochastic_event_set[0].name}
        plot_args.update(fixed_plot_args)
        ax = plot_ecdf(*ecdf(sim_counts), observation_count, catalog=observation, plot_args=plot_args, filename=filename)

        # annotate the plot with information from catalog
        ax.annotate(r'$\delta_1 = P(X \geq x) = {:.5f}$\n$\delta_2 = P(X \leq x) = {:.5f}$'
                    .format(delta_1, delta_2), xycoords='axes fraction', xy=(0.5, 0.3), fontsize=14)
        ax.set_title("CSEP2 Number Test", fontsize=14)

        # func has different return types, before release refactor and remove plotting from evaluation.
        # plotting should be separated from evaluation.
        # evaluation should return some object that can be plotted maybe with verbose option.
        return (delta_1, delta_2), ax

    if show:
        pyplot.show()

    return delta_1, delta_2


def _number_test(sim_counts, obs_count):
    """
    Direct call using ndarray. Useful for optimizing calls to multiprocessing pools.
    """
    # delta 1 prob of observation at least n_obs events given the forecast
    delta_1 = greater_equal_ecdf(sim_counts, obs_count)
    # delta 2 prob of observing at most n_obs events given the catalog
    delta_2 = less_equal_ecdf(sim_counts, obs_count)
    return delta_1, delta_2

