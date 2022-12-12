import numpy
import scipy.stats
import scipy.spatial

from csep.models import EvaluationResult
from csep.core.exceptions import CSEPCatalogException


def _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance, epsilon=1e-6):
    """ Computes delta1 and delta2 values from the Negative Binomial (NBD) number test.

    Args:
        fore_cnt (float): parameter of negative binomial distribution coming from expected value of the forecast
        obs_cnt (float): count of earthquakes observed during the testing period.
        variance (float): variance parameter of negative binomial distribution coming from historical catalog. 
        A variance value of approximately 23541 has been calculated using M5.95+ earthquakes observed worldwide from 1982 to 2013.
        epsilon (float): tolerance level to satisfy the requirements of two-sided p-value

    Returns
        result (tuple): (delta1, delta2)
    """
    var = variance
    mean = fore_cnt
    upsilon = 1.0 - ((var - mean) / var)
    tau = (mean**2 /(var - mean))
    
    delta1 = 1.0 - scipy.stats.nbinom.cdf(obs_cnt - epsilon, tau, upsilon, loc=0)
    delta2 = scipy.stats.nbinom.cdf(obs_cnt + epsilon, tau, upsilon, loc=0)

    return delta1, delta2

    
def negative_binomial_number_test(gridded_forecast, observed_catalog, variance):
    """ Computes "negative binomial N-Test" on a gridded forecast.

    Computes Number (N) test for Observed and Forecasts. Both data sets are expected to be in terms of event counts.
    We find the Total number of events in Observed Catalog and Forecasted Catalogs. Which are then employed to compute the 
    probablities of
    (i) At least no. of events (delta 1)
    (ii) At most no. of events (delta 2) assuming the negative binomial distribution.

    Args:
        gridded_forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        observed_catalog:   Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
        variance:   Variance parameter of negative binomial distribution obtained from historical catalog.            

    Returns:
        out (tuple): (delta_1, delta_2)
    """
    result = EvaluationResult()

    # observed count
    obs_cnt = observed_catalog.event_count

    # forecasts provide the expeceted number of events during the time horizon of the forecast
    fore_cnt = gridded_forecast.event_count

    epsilon = 1e-6

    # stores the actual result of the number test
    delta1, delta2 = _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance, epsilon=epsilon)
    
    # store results
    result.test_distribution = ('negative_binomial', fore_cnt)
    result.name = 'NBD N-Test'
    result.observed_statistic = obs_cnt
    result.quantile = (delta1, delta2)
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result
  
    
def binary_joint_log_likelihood_ndarray(forecast, catalog):
    """ Computes Bernoulli log-likelihood scores, assuming that earthquakes follow a binomial distribution.
    
    Args:
        forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        catalog:    Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
    """
    # First, we mask the forecast in cells where we could find log=0.0 singularities:
    forecast_masked = numpy.ma.masked_where(forecast.ravel() <= 0.0, forecast.ravel())
    # Then, we compute the log-likelihood of observing one or more events given a Poisson distribution, i.e., 1 - Pr(0)
    target_idx = numpy.nonzero(catalog.ravel())
    y = numpy.zeros(forecast_masked.ravel().shape)
    y[target_idx[0]] = 1
    first_term = y * (numpy.log(1.0 - numpy.exp(-forecast_masked.ravel())))
    # Also, we estimate the log-likelihood in cells no events are observed:
    second_term = (1-y) * (-forecast_masked.ravel().data)
    # Finally, we sum both terms to compute the joint log-likelihood score:
    return sum(first_term.data + second_term.data)
    


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
    

def _binary_likelihood_test(forecast_data, observed_data, num_simulations=1000, random_numbers=None, 
                              seed=None, use_observed_counts=True, verbose=True, normalize_likelihood=False):
    """  Computes binary conditional-likelihood test from CSEP using an efficient simulation based approach.
    
    Args:
        forecast_data (numpy.ndarray): nd array where [:, -1] are the magnitude bins.
        observed_data (numpy.ndarray): same format as observation.
        num_simulations: default number of simulations to use for likelihood based simulations
        seed: used for reproducibility of the prng
        random_numbers (numpy.ndarray): can supply an explicit list of random numbers, primarily used for software testing
        use_observed_counts (bool): if true, will simulate catalogs using the observed events, if false will draw from poisson 
        distribution
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
    simulated_ll = []
    n_active_cells = len(numpy.unique(numpy.nonzero(observed_data.ravel())))
    n_fore = numpy.sum(forecast_data)
    expected_forecast_count = int(n_active_cells)

    # main simulation step in this loop
    for idx in range(num_simulations):
        if use_observed_counts:
            num_cells_to_simulate = int(n_active_cells)
    
        if random_numbers is None:
            sim_fore = _simulate_catalog(num_cells_to_simulate, sampling_weights, sim_fore)
        else:
            sim_fore = _simulate_catalog(num_cells_to_simulate, sampling_weights, sim_fore,
                                         random_numbers=random_numbers[idx,:])

        # compute joint log-likelihood
        current_ll = binary_joint_log_likelihood_ndarray(forecast_data.data, sim_fore)
        
        # append to list of simulated log-likelihoods
        simulated_ll.append(current_ll)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')

    # observed joint log-likelihood
    obs_ll = binary_joint_log_likelihood_ndarray(forecast_data.data, observed_data)
        
    # quantile score
    qs = numpy.sum(simulated_ll <= obs_ll) / num_simulations

    # float, float, list
    return qs, obs_ll, simulated_ll
 
    
def binary_spatial_test(gridded_forecast, observed_catalog, num_simulations=1000, seed=None, random_numbers=None, verbose=False):
    """  Performs the binary spatial test on the Forecast using the Observed Catalogs.
    
    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.
    
    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.
        
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    gridded_catalog_data = observed_catalog.spatial_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binary_likelihood_test(
        gridded_forecast.spatial_counts(),
        gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose,
        normalize_likelihood=True
    )

    
# populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary S-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    try:
        result.min_mw = numpy.min(gridded_forecast.magnitudes)
    except AttributeError:
        result.min_mw = -1
    return result
    
    
def binary_conditional_likelihood_test(gridded_forecast, observed_catalog, num_simulations=1000, seed=None, random_numbers=None, verbose=False):
    """ Performs the binary conditional likelihood test on Gridded Forecast using an Observed Catalog.

    Normalizes the forecast so the forecasted rate are consistent with the observations. This modification
    eliminates the strong impact differences in the number distribution have on the forecasted rates.

    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """
        
    # grid catalog onto spatial grid
    try:
        _ = observed_catalog.region.magnitudes
    except CSEPCatalogException:
        observed_catalog.region = gridded_forecast.region
    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binary_likelihood_test(
        gridded_forecast.data,
        gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose,
        normalize_likelihood=False
    )

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary CL-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result
     
    
def matrix_binary_t_test(target_event_rates1, target_event_rates2, n_obs, n_f1, n_f2, catalog, alpha=0.05):
    """ Computes binary T test statistic by comparing two target event rate distributions.

    We compare Forecast from Model 1 and with Forecast of Model 2. Information Gain per Active Bin (IGPA) is computed, which is then 
    employed to compute T statistic. Confidence interval of Information Gain can be computed using T_critical. For a complete 
    explanation see Rhoades, D. A., et al., (2011). Efficient testing of earthquake forecasting models. Acta Geophysica, 59(4), 
    728-747. doi:10.2478/s11600-011-0013-5, and Bayona J.A. et al., (2022). Prospective evaluation of multiplicative hybrid earthquake 
    forecasting models in California. doi: 10.1093/gji/ggac018.
    
    Args:
        target_event_rates1 (numpy.ndarray): nd-array storing target event rates
        target_event_rates2 (numpy.ndarray): nd-array storing target event rates
        n_obs (float, int, numpy.ndarray): number of observed earthquakes, should be whole number and >= zero.
        n_f1 (float): Total number of forecasted earthquakes by Model 1
        n_f2 (float): Total number of forecasted earthquakes by Model 2
        catalog: csep.core.catalogs.Catalog
        alpha (float): tolerance level for the type-i error rate of the statistical test

    Returns:
        out (dict): relevant statistics from the t-test
    """
    # Some Pre Calculations -  Because they are being used repeatedly.
    N_p = n_obs  
    N = len(np.unique(np.nonzero(catalog.spatial_magnitude_counts().ravel()))) # Number of active bins
    N1 = n_f1  
    N2 = n_f2  
    X1 = numpy.log(target_event_rates1)  # Log of every element of Forecast 1
    X2 = numpy.log(target_event_rates2)  # Log of every element of Forecast 2
    

    # Information Gain, using Equation (17)  of Rhoades et al. 2011
    information_gain = (numpy.sum(X1 - X2) - (N1 - N2)) / N

    # Compute variance of (X1-X2) using Equation (18)  of Rhoades et al. 2011
    first_term = (numpy.sum(numpy.power((X1 - X2), 2))) / (N - 1)
    second_term = numpy.power(numpy.sum(X1 - X2), 2) / (numpy.power(N, 2) - N)
    forecast_variance = first_term - second_term

    forecast_std = numpy.sqrt(forecast_variance)
    t_statistic = information_gain / (forecast_std / numpy.sqrt(N))

    # Obtaining the Critical Value of T from T distribution.
    df = N - 1
    t_critical = scipy.stats.t.ppf(1 - (alpha / 2), df)  # Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

    # Computing Information Gain Interval.
    ig_lower = information_gain - (t_critical * forecast_std / numpy.sqrt(N))
    ig_upper = information_gain + (t_critical * forecast_std / numpy.sqrt(N))

    # If T value greater than T critical, Then both Lower and Upper Confidence Interval limits will be greater than Zero.
    # If above Happens, Then It means that Forecasting Model 1 is better than Forecasting Model 2.
    return {'t_statistic': t_statistic,
            't_critical': t_critical,
            'information_gain': information_gain,
            'ig_lower': ig_lower,
            'ig_upper': ig_upper}
    

def binary_paired_t_test(forecast, benchmark_forecast, observed_catalog, alpha=0.05, scale=False):
    """ Computes the binary t-test for gridded earthquake forecasts.

    This score is positively oriented, meaning that positive values of the information gain indicate that the
    forecast is performing better than the benchmark forecast

    Args:
        forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        benchmark_forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude 
        column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test
        scale (bool): if true, scale forecasted rates down to a single day

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # needs some pre-processing to put the forecasts in the context that is required for the t-test. this is different
    # for cumulative forecasts (eg, multiple time-horizons) and static file-based forecasts.
    target_event_rate_forecast1p, n_fore1 = forecast.target_event_rates(observed_catalog, scale=scale)
    target_event_rate_forecast2p, n_fore2 = benchmark_forecast.target_event_rates(observed_catalog, scale=scale)
    
    target_event_rate_forecast1 = forecast.data.ravel()[np.unique(np.nonzero(observed_catalog.spatial_magnitude_counts().ravel()))]
    target_event_rate_forecast2 = benchmark_forecast.data.ravel()[np.unique(np.nonzero(observed_catalog.spatial_magnitude_counts().
    ravel()))]

    # call the primative version operating on ndarray
    out = matrix_binary_t_test(target_event_rate_forecast1, target_event_rate_forecast2, observed_catalog.event_count, n_fore1, n_fore2,
                          observed_catalog,
                          alpha=alpha)

    # storing this for later
    result = EvaluationResult()
    result.name = 'binary paired T-Test'
    result.test_distribution = (out['ig_lower'], out['ig_upper'])
    result.observed_statistic = out['information_gain']
    result.quantile = (out['t_statistic'], out['t_critical'])
    result.sim_name = (forecast.name, benchmark_forecast.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = np.min(forecast.magnitudes)
    return result
