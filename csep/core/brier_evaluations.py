import numpy
import numpy as np
from scipy.stats import poisson

from csep.models import EvaluationResult
from csep.core.exceptions import CSEPCatalogException


def _brier_score_ndarray(forecast, observations):
    """ Computes the brier (binary) score for spatial-magnitude cells
    using the formula:

    Q(Lambda, Sigma) = 1/N sum_{i=1}^N (Lambda_i - Ind(Sigma_i > 0 ))^2

    where Lambda is the forecast array, Sigma is the observed catalog, N the
    number of spatial-magnitude cells and Ind is the indicator function, which
    is 1 if Sigma_i > 0 and 0 otherwise.

    Args:
        forecast: 2d array of forecasted rates
        observations: 2d array of observed counts
    Returns
        brier: float, brier score
    """

    prob_success = 1 - poisson.cdf(0, forecast)
    brier_cell = np.square(prob_success.ravel() - (observations.ravel() > 0))
    brier = -2 * brier_cell.sum()

    for n_dim in observations.shape:
        brier /= n_dim
    return brier


def _simulate_catalog(sim_cells, sampling_weights, random_numbers=None):
    """
    Simulates a catalog by sampling from the sampling_weights array.
    Identical to binomial_evaluations._simulate_catalog

    Args:
        sim_cells:
        sampling_weights:
        random_numbers:

    Returns:

    """
    sim_fore = numpy.zeros(sampling_weights.shape)

    if random_numbers is None:
        # Reset simulation array to zero, but don't reallocate
        sim_fore.fill(0)
        num_active_cells = 0
        while num_active_cells < sim_cells:
            random_num = numpy.random.uniform(0,1)
            loc = numpy.searchsorted(sampling_weights, random_num,
                                     side='right')
            if sim_fore[loc] == 0:
                sim_fore[loc] = 1
                num_active_cells += 1
    else:
        # Find insertion points using binary search inserting
        # to satisfy a[i-1] <= v < a[i]
        pnts = numpy.searchsorted(sampling_weights, random_numbers,
                                  side='right')
        # Create simulated catalog by adding to the original locations
        numpy.add.at(sim_fore, pnts, 1)

    assert sim_fore.sum() == sim_cells, "simulated the wrong number of events!"
    return sim_fore


def _brier_score_test(forecast_data, observed_data, num_simulations=1000,
                      random_numbers=None, seed=None, verbose=True):
    """  Computes the Brier consistency test conditional on the total observed
    number of events

    Args:
        forecast_data: 2d array of forecasted rates for spatial_magnitude cells
        observed_data: 2d array of a catalog resampled to spatial_magnitude
            cells
        num_simulations: number of synthetic catalog simulations
        random_numbers: numpy array of random numbers to use for simulation
        seed: seed for random number generator
        verbose: print status updates



    """
    # Array-masking that avoids log singularities:
    forecast_data = numpy.ma.masked_where(forecast_data <= 0.0, forecast_data)

    # set seed for the likelihood test
    if seed is not None:
        numpy.random.seed(seed)

    # used to determine where simulated earthquake should
    # be placed, by definition of cumsum these are sorted
    sampling_weights = (numpy.cumsum(forecast_data.ravel()) /
                        numpy.sum(forecast_data))

    # data structures to store results
    simulated_brier = []
    n_active_cells = len(numpy.unique(numpy.nonzero(observed_data.ravel())))
    num_cells_to_simulate = int(n_active_cells)

    # main simulation step in this loop
    for idx in range(num_simulations):
        if random_numbers is None:
            sim_fore = _simulate_catalog(num_cells_to_simulate,
                                         sampling_weights)
        else:
            sim_fore = _simulate_catalog(num_cells_to_simulate,
                                         sampling_weights,
                                         random_numbers=random_numbers[idx, :])

        # compute Brier score
        current_brier = _brier_score_ndarray(forecast_data.data, sim_fore)

        # append to list of simulated Brier score
        simulated_brier.append(current_brier)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')

    obs_brier = _brier_score_ndarray(forecast_data.data, observed_data)
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
    """
    Performs the Brier conditional test on a Gridded Forecast using an
    Observed Catalog. Normalizes the forecast so the forecasted rate are
    consistent with the observations. This modification eliminates the strong
    impact differences in the number distribution have on the forecasted rates.
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

