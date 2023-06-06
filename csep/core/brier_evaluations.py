import numpy
import scipy.stats
import scipy.spatial

from csep.models import EvaluationResult
from csep.core.exceptions import CSEPCatalogException


def bier_score_ndarray(forecast, observations):
    """ Computes the brier score
    """

    observations = (observations >= 1).astype(int)
    prob_success = 1 - scipy.stats.poisson.cdf(0, forecast)
    brier = []

    for p, o in zip(prob_success.ravel(), observations.ravel()):

        if o == 0:
            brier.append(-2 * p ** 2)
        else:
            brier.append(-2 * (p - 1) ** 2)
    brier = numpy.sum(brier)

    for n_dim in observations.shape:
        brier /= n_dim

    return brier


def _simulate_catalog(sim_cells, sampling_weights, sim_fore, random_numbers=None):
    # Modified this code to generate simulations in a way that every cell gets one earthquake
    # Generate uniformly distributed random numbers in [0,1), this
    if random_numbers is None:
        # Reset simulation array to zero, but don't reallocate
        sim_fore.fill(0)
        num_active_cells = 0
        while num_active_cells < sim_cells:
            random_num = numpy.random.uniform(0,1)
            loc = numpy.searchsorted(sampling_weights, random_num, side='right')
            if sim_fore[loc] == 0:
                sim_fore[loc] = 1
                num_active_cells = num_active_cells + 1
    else:
        # Find insertion points using binary search inserting to satisfy a[i-1] <= v < a[i]
        pnts = numpy.searchsorted(sampling_weights, random_numbers, side='right')
        # Create simulated catalog by adding to the original locations
        numpy.add.at(sim_fore, pnts, 1)

    assert sim_fore.sum() == sim_cells, "simulated the wrong number of events!"
    return sim_fore


def _brier_score_test(forecast_data, observed_data, num_simulations=1000, random_numbers=None,
                            seed=None, use_observed_counts=True, verbose=True):
    """  Computes binary conditional-likelihood test from CSEP using an efficient simulation based approach.

    Args:

    """

    # Array-masking that avoids log singularities:
    forecast_data = numpy.ma.masked_where(forecast_data <= 0.0, forecast_data)

    # set seed for the likelihood test
    if seed is not None:
        numpy.random.seed(seed)

    # used to determine where simulated earthquake should be placed, by definition of cumsum these are sorted
    sampling_weights = numpy.cumsum(forecast_data.ravel()) / numpy.sum(forecast_data)

    # data structures to store results
    sim_fore = numpy.zeros(sampling_weights.shape)
    simulated_brier = []
    n_active_cells = len(numpy.unique(numpy.nonzero(observed_data.ravel())))

    # main simulation step in this loop
    for idx in range(num_simulations):
        if use_observed_counts:
            num_cells_to_simulate = int(n_active_cells)

        if random_numbers is None:
            sim_fore = _simulate_catalog(num_cells_to_simulate,
                                         sampling_weights,
                                         sim_fore)
        else:
            sim_fore = _simulate_catalog(num_cells_to_simulate,
                                         sampling_weights,
                                         sim_fore,
                                         random_numbers=random_numbers[idx, :])

        # compute Brier score
        current_brier = bier_score_ndarray(forecast_data.data, sim_fore)

        # append to list of simulated Brier score
        simulated_brier.append(current_brier)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')

    # observed Brier score
    obs_brier = bier_score_ndarray(forecast_data.data, observed_data)

    # quantile score
    qs = numpy.sum(simulated_brier <= obs_brier) / num_simulations

    # float, float, list
    return qs, obs_brier, simulated_brier


def brier_score_test(gridded_forecast,
                     observed_catalog,
                     num_simulations=1000,
                     seed=None,
                     random_numbers=None,
                     verbose=False):
    """ Performs the Brier conditional test on Gridded Forecast using an Observed Catalog.

    Normalizes the forecast so the forecasted rate are consistent with the observations. This modification
    eliminates the strong impact differences in the number distribution have on the forecasted rates.

    """

    # grid catalog onto spatial grid
    try:
        _ = observed_catalog.region.magnitudes
    except CSEPCatalogException:
        observed_catalog.region = gridded_forecast.region
    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_brier, simulated_brier = _brier_score_test(
        gridded_forecast.data,
        gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose
    )

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_brier
    result.name = 'Brier score-Test'
    result.observed_statistic = obs_brier
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result

