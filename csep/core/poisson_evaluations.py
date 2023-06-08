#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import scipy.stats
import scipy.spatial
import warnings

from csep.models import EvaluationResult
from csep.utils.stats import poisson_joint_log_likelihood_ndarray
from csep.core.exceptions import CSEPCatalogException


def paired_t_test(forecast, benchmark_forecast, observed_catalog,
                  alpha=0.05, scale=False):
    """ Computes the t-test for gridded earthquake forecasts.

    This score is positively oriented, meaning that positive values of the information gain indicate that the
    forecast is performing better than the benchmark forecast.

    Args:
        forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        benchmark_forecast (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test
        scale (bool): if true, scale forecasted rates down to a single day

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # needs some pre-processing to put the forecasts in the context that is required for the t-test. this is different
    # for cumulative forecasts (eg, multiple time-horizons) and static file-based forecasts.
    target_event_rate_forecast1, n_fore1 = forecast.target_event_rates(
        observed_catalog, scale=scale)
    target_event_rate_forecast2, n_fore2 = benchmark_forecast.target_event_rates(
        observed_catalog, scale=scale)

    # call the primative version operating on ndarray
    out = _t_test_ndarray(target_event_rate_forecast1,
                          target_event_rate_forecast2,
                          observed_catalog.event_count,
                          n_fore1, n_fore2, alpha=alpha)

    # storing this for later
    result = EvaluationResult()
    result.name = 'Paired T-Test'
    result.test_distribution = (out['ig_lower'], out['ig_upper'])
    result.observed_statistic = out['information_gain']
    result.quantile = (out['t_statistic'], out['t_critical'])
    result.sim_name = (forecast.name, benchmark_forecast.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(forecast.magnitudes)
    return result


def w_test(gridded_forecast1, gridded_forecast2, observed_catalog,
           scale=False):
    """ Calculate the Single Sample Wilcoxon signed-rank test between two gridded forecasts.

    This test allows to test the null hypothesis that the median of Sample (X1(i)-X2(i)) is equal to a (N1-N2) / N_obs.
    where, N1, N2 = Sum of expected values of Forecast_1 and Forecast_2, respectively.

    The Wilcoxon signed-rank test tests the null hypothesis that difference of Xi and Yi come from the same distribution.
    In particular, it tests whether the distribution of the differences is symmetric around given mean.

    Parameters

    Args:
        gridded_forecast1: Forecast of a model_1 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin
                It can be anything greater than zero

        gridded_forecast2: Forecast of model_2 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin
                It can be anything greater than zero

        observation: Observed (Grided) seismicity (Numpy Array):
                An Observation has to be observed seismicity in each Bin
                It has to be a either zero or positive integer only (No Floating Point)

    Returns
        out: csep.core.evaluations.EvaluationResult
    """

    # needs some pre-processing to put the forecasts in the context that is required for the t-test. this is different
    # for cumulative forecasts (eg, multiple time-horizons) and static file-based forecasts.
    target_event_rate_forecast1, _ = gridded_forecast1.target_event_rates(
        observed_catalog, scale=scale)
    target_event_rate_forecast2, _ = gridded_forecast2.target_event_rates(
        observed_catalog, scale=scale)

    N = observed_catalog.event_count  # Sum of all the observed earthquakes
    N1 = gridded_forecast1.event_count  # Total number of Forecasted earthquakes by Model 1
    N2 = gridded_forecast2.event_count  # Total number of Forecasted earthquakes by Model 2
    X1 = numpy.log(
        target_event_rate_forecast1)  # Log of every element of Forecast 1
    X2 = numpy.log(
        target_event_rate_forecast2)  # Log of every element of Forecast 2

    # this ratio is the same as long as we scale all the forecasts and catalog rates by the same value
    median_value = (N1 - N2) / N

    diff = X1 - X2

    # w_test is One Sample Wilcoxon Signed Rank Test. It accepts the data only in 1D array.
    x = diff.ravel()  # Converting 2D Difference to 1D

    w_test_dic = _w_test_ndarray(x, median_value)

    # configure test result
    result = EvaluationResult()
    result.name = 'W-Test'
    result.test_distribution = 'normal'
    result.observed_statistic = w_test_dic['z_statistic']
    result.quantile = w_test_dic['probability']
    result.sim_name = (gridded_forecast1.name, gridded_forecast2.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast1.magnitudes)
    return result


def number_test(gridded_forecast, observed_catalog):
    """Computes "N-Test" on a gridded forecast.
    author: @asim

    Computes Number (N) test for Observed and Forecasts. Both data sets are expected to be in terms of event counts.
    We find the Total number of events in Observed Catalog and Forecasted Catalogs. Which are then employed to compute the probablities of
    (i) At least no. of events (delta 1)
    (ii) At most no. of events (delta 2) assuming the poissonian distribution.

    Args:
        observation: Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
        forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero

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
    delta1, delta2 = _number_test_ndarray(fore_cnt, obs_cnt, epsilon=epsilon)

    # store results
    result.test_distribution = ('poisson', fore_cnt)
    result.name = 'Poisson N-Test'
    result.observed_statistic = obs_cnt
    result.quantile = (delta1, delta2)
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def conditional_likelihood_test(gridded_forecast, observed_catalog,
                                num_simulations=1000, seed=None,
                                random_numbers=None, verbose=False):
    """Performs the conditional likelihood test on Gridded Forecast using an Observed Catalog.

    This test normalizes the forecast so the forecasted rate are consistent with the observations. This modification
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
    qs, obs_ll, simulated_ll = _poisson_likelihood_test(gridded_forecast.data,
                                                        gridded_catalog_data,
                                                        num_simulations=num_simulations,
                                                        seed=seed,
                                                        random_numbers=random_numbers,
                                                        use_observed_counts=True,
                                                        verbose=verbose,
                                                        normalize_likelihood=False)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Poisson CL-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def poisson_spatial_likelihood(forecast, catalog):
    """
    This function computes the observed log-likehood score obtained by a gridded forecast in each cell, given a
    seismicity catalog. In this case, we assume a Poisson distribution of earthquakes, so that the likelihood of
    observing an event w given the expected value x in each cell is:
    poll = -x + wlnx - ln(w!)
    
    Args:
    	forecast: gridded forecast
    	catalog: observed catalog
    
    Returns:
    	poll: Poisson-based log-likelihood scores obtained by the forecast in each spatial cell.
    
    Notes:
    	log(w!) = 0
    	factorial(n) = loggamma(n+1)
    """

    scale = catalog.event_count / forecast.event_count

    first_term = -forecast.spatial_counts() * scale
    second_term = catalog.spatial_counts() * numpy.log(
        forecast.spatial_counts() * scale)
    third_term = -scipy.special.loggamma(catalog.spatial_counts() + 1)

    poll = first_term + second_term + third_term

    return poll


def binary_spatial_likelihood(forecast, catalog):
    """
    This function computes log-likelihood scores (bills), using a binary likelihood distribution of earthquakes.
    For this aim, we need an input variable 'forecast' and an variable 'catalog'
    
    This function computes the observed log-likehood score obtained by a gridded forecast in each cell, given a
    seismicity catalog. In this case, we assume a binary distribution of earthquakes, so that the likelihood of
    observing an event w given the expected value x in each cell is:'
    bill = (1-X) * ln(exp(-λ)) + X * ln(1 - exp(-λ)), with X=1 if earthquake and X=0 if no earthquake.
    
    Args:
    	forecast: gridded forecast
    	catalog: observed catalog
    
    Returns:
    bill: Binary-based log-likelihood scores obtained by the forecast in each spatial cell.
    """

    scale = catalog.event_count / forecast.event_count
    target_idx = numpy.nonzero(catalog.spatial_counts())
    X = numpy.zeros(forecast.spatial_counts().shape)
    X[target_idx[0]] = 1

    # First, we estimate the log-likelihood in cells where no events are observed:
    first_term = (1 - X) * (-forecast.spatial_counts() * scale)

    # Then, we compute the log-likelihood of observing one or more events given a Poisson distribution, i.e., 1 - Pr(0):
    second_term = X * (
        numpy.log(1.0 - numpy.exp(-forecast.spatial_counts() * scale)))

    # Finally, we sum both terms to compute log-likelihood score in each spatial cell:
    bill = first_term + second_term

    return bill


def magnitude_test(gridded_forecast, observed_catalog, num_simulations=1000,
                   seed=None, random_numbers=None,
                   verbose=False):
    """
    Performs the Magnitude Test on a Gridded Forecast using an observed catalog.

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
    gridded_catalog_data = observed_catalog.magnitude_counts(
        mag_bins=gridded_forecast.magnitudes)

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _poisson_likelihood_test(
        gridded_forecast.magnitude_counts(), gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose,
        normalize_likelihood=True)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Poisson M-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def spatial_test(gridded_forecast, observed_catalog, num_simulations=1000,
                 seed=None, random_numbers=None,
                 verbose=False):
    """
    Performs the Spatial Test on the Forecast using the Observed Catalogs.

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

    gridded_catalog_data = observed_catalog.spatial_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _poisson_likelihood_test(
        gridded_forecast.spatial_counts(), gridded_catalog_data,
        num_simulations=num_simulations,
        seed=seed,
        random_numbers=random_numbers,
        use_observed_counts=True,
        verbose=verbose,
        normalize_likelihood=True)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Poisson S-Test'
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


def likelihood_test(gridded_forecast, observed_catalog, num_simulations=1000,
                    seed=None, random_numbers=None,
                    verbose=False):
    """
    Performs the likelihood test on Gridded Forecast using an Observed Catalog.

    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation.
                               injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    # grid catalog onto spatial grid
    try:
        _ = observed_catalog.region.magnitudes
    except CSEPCatalogException:
        observed_catalog.region = gridded_forecast.region

    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog and forecast
    qs, obs_ll, simulated_ll = _poisson_likelihood_test(gridded_forecast.data,
                                                        gridded_catalog_data,
                                                        num_simulations=num_simulations,
                                                        seed=seed,
                                                        random_numbers=random_numbers,
                                                        use_observed_counts=False,
                                                        verbose=verbose,
                                                        normalize_likelihood=False)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Poisson L-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def _number_test_ndarray(fore_cnt, obs_cnt, epsilon=1e-6):
    """ Computes delta1 and delta2 values from the csep1 number test.

    Args:
        fore_cnt (float): parameter of poisson distribution coming from expected value of the forecast
        obs_cnt (float): count of earthquakes observed during the testing period.
        epsilon (float): tolerance level to satisfy the requirements of two-sided p-value

    Returns
        result (tuple): (delta1, delta2)
    """
    delta1 = 1.0 - scipy.stats.poisson.cdf(obs_cnt - epsilon, fore_cnt)
    delta2 = scipy.stats.poisson.cdf(obs_cnt + epsilon, fore_cnt)
    return delta1, delta2


def _t_test_ndarray(target_event_rates1, target_event_rates2, n_obs, n_f1,
                    n_f2, alpha=0.05):
    """ Computes T test statistic by comparing two target event rate distributions.

    We compare Forecast from Model 1 and with Forecast of Model 2. Information Gain is computed, which is then employed
    to compute T statistic. Confidence interval of Information Gain can be computed using T_critical. For a complete explanation
    see Rhoades, D. A., et al., (2011). Efficient testing of earthquake forecasting models. Acta Geophysica, 59(4), 728-747.
    doi:10.2478/s11600-011-0013-5

    Args:
        target_event_rates1 (numpy.ndarray): nd-array storing target event rates
        target_event_rates2 (numpy.ndarray): nd-array storing target event rates
        n_obs (float, int, numpy.ndarray): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test

    Returns:
        out (dict): relevant statistics from the t-test

    """
    # Some Pre Calculations -  Because they are being used repeatedly.
    N = n_obs  # Total number of observed earthquakes
    N1 = n_f1  # Total number of forecasted earthquakes by Model 1
    N2 = n_f2  # Total number of forecasted earthquakes by Model 2
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
    t_critical = scipy.stats.t.ppf(1 - (alpha / 2),
                                   df)  # Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

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


def _w_test_ndarray(x, m=0):
    """ Calculate the Single Sample Wilcoxon signed-rank test for an ndarray.

    This method is based on collecting a number of samples from a population with unknown median, m.
    The Wilcoxon One Sample Signed-Rank test is the non parametric version of the t-test.
    It is based on ranks and because of that, the location parameter is not here the mean but the median.
    This test allows to test the null hypothesis that the sample median is equal to a given value provided by the user.
    If we designate m to be the assumed median of the sample:
    Null hypothesis (simplified): The population from which the data were sampled is symmetric about the Given value (m).
    Alternative hypothesis (simplified, two-sided): The population from which the data were sampled is not symmetric around m.

    Args:
        x:   1D vector of paired differences.
        m:   Designated median value.

    Returns:
        dict: {'z_statistic': Value of Z statistic, considering two-side test,
               'probablity': Probablity value }
    """
    # compute median differences
    d = x - m

    # remove zero values
    d = numpy.compress(numpy.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Sample size too small for normal approximation.")

    # compute ranks
    r = scipy.stats.rankdata(abs(d))
    r_plus = numpy.sum((d > 0) * r, axis=0)
    r_minus = numpy.sum((d < 0) * r, axis=0)

    # for "two-sided", choose minimum of both
    t = min(r_plus, r_minus)

    # Correction to be introduced
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    replist, repnum = scipy.stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = numpy.sqrt(se / 24)

    # compute statistic and p-value using normal approximation
    #  z = (T - mn - d) / se Continuity correction. But We are not considering continuity correction.
    z = (t - mn) / se

    # 2, is multiplied for "two-sided" distribution
    prob = 2. * scipy.stats.distributions.norm.sf(abs(z))

    # Accept the NULL Hypothesis [Median(Xi-Yi) = Given value]. If probability is greater than 0.05
    # If Probability is smaller than 0.05, Reject the NULL Hypothesis, that Median(Xi-Yi) != Given Value
    w_test_eval = {'z_statistic': z,
                   'probability': prob}

    return w_test_eval


def _simulate_catalog(num_events, sampling_weights, sim_fore,
                      random_numbers=None):
    # generate uniformly distributed random numbers in [0,1), this
    if random_numbers is None:
        random_numbers = numpy.random.rand(num_events)
    else:
        # TODO: ensure that random numbers are all between 0 and 1.
        pass

    # reset simulation array to zero, but don't reallocate
    sim_fore.fill(0)

    # find insertion points using binary search inserting to satisfy a[i-1] <= v < a[i]
    pnts = numpy.searchsorted(sampling_weights, random_numbers, side='right')

    # create simulated catalog by adding to the original locations
    numpy.add.at(sim_fore, pnts, 1)
    assert sim_fore.sum() == num_events, "simulated the wrong number of events!"

    return sim_fore


def _poisson_likelihood_test(forecast_data, observed_data,
                             num_simulations=1000, random_numbers=None,
                             seed=None, use_observed_counts=True, verbose=True,
                             normalize_likelihood=False):
    """
	Computes the likelihood-test from CSEP using an efficient simulation based approach.
	Args:
	    forecast_data (numpy.ndarray): nd array where [:, -1] are the magnitude bins.
	    observed_data (numpy.ndarray): same format as observation.
	    num_simulations: default number of simulations to use for likelihood based simulations
    	seed: used for reproducibility of the prng
	    random_numbers (numpy.ndarray): can supply an explicit list of random numbers, primarily used for software testing
	    use_observed_counts (bool): if true, will simulate catalogs using the observed events, if false will draw from poisson distribution
	    verbose (bool): if true, write progress of test to command line
	    normalize_likelihood (bool): if true, normalize likelihood. used by deafult for magnitude and spatial tests
	"""

    # set seed for the likelihood test
    if seed is not None:
        numpy.random.seed(seed)

    # used to determine where simulated earthquake should be placed, by definition of cumsum these are sorted
    sampling_weights = numpy.cumsum(forecast_data.ravel()) / numpy.sum(
        forecast_data)

    # data structures to store results
    sim_fore = numpy.zeros(sampling_weights.shape)
    simulated_ll = []

    # properties of observations and forecasts
    n_obs = numpy.sum(observed_data)
    n_fore = numpy.sum(forecast_data)

    expected_forecast_count = numpy.sum(forecast_data)
    log_bin_expectations = numpy.log(forecast_data.ravel())
    # used for conditional-likelihood, magnitude, and spatial tests to normalize the rate-component of the forecasts
    if use_observed_counts and normalize_likelihood:
        scale = n_obs / n_fore
        expected_forecast_count = int(n_obs)
        log_bin_expectations = numpy.log(forecast_data.ravel() * scale)

    # gets the 1d indices to bins that contain target events, these indexes perform copies and not views into the array
    target_idx = numpy.nonzero(observed_data.ravel())

    # note for performance: these operations perform copies
    observed_data_nonzero = observed_data.ravel()[target_idx]
    target_event_forecast = log_bin_expectations[
                                target_idx] * observed_data_nonzero

    # main simulation step in this loop
    for idx in range(num_simulations):
        if use_observed_counts:
            num_events_to_simulate = int(n_obs)
        else:
            num_events_to_simulate = int(
                numpy.random.poisson(expected_forecast_count))

        if random_numbers is None:
            sim_fore = _simulate_catalog(num_events_to_simulate,
                                         sampling_weights, sim_fore)
        else:
            sim_fore = _simulate_catalog(num_events_to_simulate,
                                         sampling_weights, sim_fore,
                                         random_numbers=random_numbers[idx, :])

        # compute joint log-likelihood from simulation by leveraging that only cells with target events contribute to likelihood
        sim_target_idx = numpy.nonzero(sim_fore)
        sim_obs_nonzero = sim_fore[sim_target_idx]
        sim_target_event_forecast = log_bin_expectations[
                                        sim_target_idx] * sim_obs_nonzero

        # compute joint log-likelihood
        current_ll = poisson_joint_log_likelihood_ndarray(
            sim_target_event_forecast, sim_obs_nonzero,
            expected_forecast_count)

        # append to list of simulated log-likelihoods
        simulated_ll.append(current_ll)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')

    # observed joint log-likelihood
    obs_ll = poisson_joint_log_likelihood_ndarray(target_event_forecast,
                                                  observed_data_nonzero,
                                                  expected_forecast_count)

    # quantile score
    qs = numpy.sum(simulated_ll <= obs_ll) / num_simulations

    # float, float, list
    return qs, obs_ll, simulated_ll
