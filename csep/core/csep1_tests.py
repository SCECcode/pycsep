#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:06:58 2020


@author: khawaja
"""
import numpy
import scipy.stats

from csep.core.evaluations import EvaluationResult
from csep.core.exceptions import CSEPValueException
from csep.utils.stats import _poisson_log_likelihood


def csep1_t_test(gridded_forecast1, gridded_forecast2, observed_catalog, alpha=0.05):
    """

    Args:
        gridded_forecast_1 (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        gridded_forecast_2 (csep.core.forecasts.GriddedForecast): nd-array storing gridded rates, axis=-1 should be the magnitude column
        observed_catalog (csep.core.catalogs.AbstractBaseCatalog): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # some defensive programming
    if gridded_forecast1.magnitudes != gridded_forecast2.magnitudes:
        raise CSEPValueException("Magnitude ranges must match on forecasts!")

    if gridded_forecast1.data.shape != gridded_forecast2.data.shape:
        raise CSEPValueException("Forecast shapes should match!")

    # needs some pre-processing to put the forecasts in the context that is required for the t-test. this is different
    # for cumulative forecasts (eg, multiple time-horizons) and static file-based forecasts.

    # call the primative version operating on ndarray
    out = csep1_t_test_ndarray(gridded_forecast1.data, gridded_forecast2.data, observed_catalog.event_count, alpha=alpha)

    # storing this for later
    result = EvaluationResult()
    result.name = 'T-Test'
    result.test_distribution = (out['ig_lower'], out['ig_upper'])
    result.observed_statistic = out['information_gain']
    result.quantile = (out['t_stat'], out['t_crit'])
    result.sim_name = (gridded_forecast1.name, gridded_forecast2.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast1.magnitudes)
    return result


def csep1_t_test_ndarray(gridded_forecast_data1, gridded_forecast_data2, n_obs, alpha=0.05):
    """ Computes T test statistic by comparing two gridded forecasts specified as numpy arrays.

    We compare Forecast from Model 1 and with Forecast of Model 2. Information Gain is computed, which is then employed
    to compute T statistic. Confidence interval of Information Gain can be computed using T_critical. For a complete explanation
    see Rhoades, D. A., et al., (2011). Efficient testing of earthquake forecasting models. Acta Geophysica, 59(4), 728-747.
    doi:10.2478/s11600-011-0013-5

    Args:
        gridded_forecast_1 (numpy.ndarray): nd-array storing gridded rates
        gridded_forecast_2 (numpy.ndarray): nd-array storing gridded rates
        n_obs (float, int, numpy.ndarray): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test

    Returns:
        out (dict): relevant statistics from the t-test

    """
    # Some Pre Calculations -  Because they are being used repeatedly.
    N = n_obs  # Total number of observed earthquakes
    N1 = numpy.sum(gridded_forecast_data1)  # Total number of forecasted earthquakes by Model 1
    N2 = numpy.sum(gridded_forecast_data2)  # Total number of forecasted earthquakes by Model 2
    X1 = numpy.log(gridded_forecast_data1)  # Log of every element of Forecast 1
    X2 = numpy.log(gridded_forecast_data2)  # Log of every element of Forecast 2

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
    return {'t_statistic': t_statistic, 't_critical': t_critical, 'information_gain': information_gain,
            'ig_lower': ig_lower, 'ig_upper': ig_upper}


def csep1_w_test_ndarray(x, m=0):
    
    """
    Calculate the Single Sample Wilcoxon signed-rank test.
    This method is based on collecting a number of samples from a population with unknown median, m.   
    The Wilcoxon One Sample Signed-Rank test is the non parametric version of the t-test.
    It is based on ranks and because of that, the location parameter is not here the mean but the median.
    This test allows to test the null hypothesis that the sample median is equal to a given value provided by the user.
    If we designate m to be the assumed median of the sample:
    Null hypothesis (simplified): The population from which the data were sampled is symmetric about the Given value (m).
    Alternative hypothesis (simplified, two-sided): The population from which the data were sampled is not symmetric around m. 
    
    Parameters
       
    Args
    x:   1D vector. For CSEP it has to be [log(forecat_1) - log(forecast_2)]
                An Observation has to be observed seismicity in each Bin
    m:   Designated mean value. For CSEP it to be (Sum of expectation of forecast_1 - Sum of expectation of forecast_2) / total no. of bins (I assume bins)
    
    Returns
    A dictionary of following elements
                {'z_statistic': Value of Z statistic, considering two-side test,
                 'probablity': Probablity value }
    """
    # compute median differences
    d = x - m

    # remove zero values
    d = numpy.compress(numpy.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        numpy.warnings.warn("Sample size too small for normal approximation.")

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

    #Accept the NULL Hypothesis [Median(Xi-Yi) = Given value]. If probability is greater than 0.05
    #If Probability is smaller than 0.05, Reject the NULL Hypothesis, that Median(Xi-Yi) != Given Value
    w_test_eval = { 'z_statistic' : z,
                    'probability': prob }

    return w_test_eval


def csep1_w_test(gridded_forecast1, gridded_forecast2, observed_catalog):
    
    """
    Calculate the Single Sample Wilcoxon signed-rank test for "log(gridded_forecast1(i))-log(gridded_forecast2(i))".
    This test allows to test the null hypothesis that the median of Sample (X1(i)-X2(i)) is equal to a (N1-N2)/nbins.
    where, N1, N2 = Sum of expected values of Forecast_1 and Forecast_2, respectively. 
    {Note: I assume N=Total No of bins here (nbins)} for Rhodes et al. 2011 (Page 741).
    The Wilcoxon signed-rank test tests the null hypothesis that difference of Xi and Yi come from the same distribution. 
    In particular, it tests whether the distribution of the differences is symmetric around given mean.
    
    Parameters
        
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be observed seismicity in each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    gridded_forecast1: Forecast of a model_1 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    gridded_forecast2: Forecast of model_2 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    model_1:    The name of Model 1, with whom the comparision will be drawn
    model_2:    The name of second model, which will be compared with Model 1
    
    Returns
    A dictionary of following elements
                {'model_name_1': Name of Model 1,
                 'model_name_2': Name of Model 2,
                 'z_statistic' : Z statistic computed between forecast 1 and forecast 2,
                 'probablity': Probablity value}
    """
    N = observed_catalog.event_count            # Sum of all the observed earthquakes
    N1 = gridded_forecast1.event_count          # Total number of Forecasted earthquakes by Model 1
    N2 = gridded_forecast2.event_count          # Total number of Forecasted earthquakes by Model 2
    X1 = numpy.log(gridded_forecast1.data)      # Log of every element of Forecast 1
    X2 = numpy.log(gridded_forecast2.data)      # Log of every element of Forecast 2
    
    median_value = (N1 - N2) / N
    
    diff = X1 - X2

    # same pre-preprocessing needs to occur for the w-test and the t-test
    
    # w_test is One Sample Wilcoxon Signed Rank Test. It accepts the data only in 1D array.
    x = diff.ravel()  # Converting 2D Difference to 1D
    
    w_test_dic = csep1_w_test_ndarray(x, median_value)

    # configure test result
    result = EvaluationResult()
    result.name = 'W-Test'
    result.test_distribution = 'normal'
    result.observed_statistic = w_test_dic['z_statistic']
    result.quantile = w_test_dic['prob']
    result.sim_name = (gridded_forecast1.name, gridded_forecast2.name)
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast1.magnitudes)
    return result


def _simulate_catalog(num_events, sampling_weights, sort_args, sim_fore, random_numbers=None, seed=None):
    if seed is not None:
        numpy.random.seed(seed)

    # generate uniformly distributed random numbers in [0,1), this
    if random_numbers is None:
        random_numbers = numpy.random.rand(num_events)

    # reset simulation array to zero, but don't reallocate
    sim_fore.fill(0)

    # find insertion points using binary search inserting to satisfy a[i-1] <= v < a[i]
    pnts = numpy.searchsorted(sampling_weights, random_numbers, sorter=sort_args, side='right')

    # create simulated catalog
    numpy.add.at(sim_fore, pnts, 1)

    return sim_fore


def poisson_likelihood_test(forecast_data, observed_data, num_simulations=1000, seed=None):
    """
    Computes the likelihood-test from CSEP using an efficient simulation based approach.

    Args:
        forecast_data (numpy.ndarray): nd array where [:, -1] are the magnitude bins.
        observed_data (numpy.ndarray): same format as observation.
    """

    # forecast_data should be event counts
    expected_forecast_count = numpy.sum(forecast_data)

    # used to determine where simulated earthquake shoudl be placed
    sampling_weights = numpy.cumsum(forecast_data.ravel()) / expected_forecast_count
    sort_args = numpy.argsort(sampling_weights)

    # data structures to store results
    sim_fore = numpy.empty(sampling_weights.shape)
    simulated_ll = []

    # observed joint log-likelihood
    obs_ll = numpy.sum(_poisson_log_likelihood(observed_data, forecast_data))

    for idx in range(num_simulations):
        sim_fore = _simulate_catalog(expected_forecast_count, sampling_weights, sort_args, sim_fore, seed=seed)

        # compute joint log-likelihood from simulation
        current_ll = numpy.sum(_poisson_log_likelihood(sim_fore, forecast_data.ravel()))

        simulated_ll.append(current_ll)

    # quantile score
    qs = numpy.sum(simulated_ll <= obs_ll) / num_simulations

    return qs, obs_ll, simulated_ll


def csep1_conditional_likelihood_test(gridded_forecast, observed_catalog, num_simulations=1000, seed=None):
    """
    Performs the conditional likelihood test on Gridded Forecast using an Observed Catalog. This test normalizes the forecast so the forecasted rate
    are consistent with the observations. This modification eliminates the strong impact differences in the number distribution have on the
    forecasted rates.

    The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
        gridded_forecast: csep.core.forecasts.MarkedGriddedDataSet
        observed_catalog: csep.core.catalogs.AbstractBaseCatalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used for reproducibility

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # storing this for later
    result = EvaluationResult()

    # scale forecast to observed event count
    scaling_factor = observed_catalog.event_count / gridded_forecast.event_count
    gridded_forecast.scale(scaling_factor)

    # grid catalog onto spatial grid
    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = poisson_likelihood_test(gridded_forecast.data, gridded_catalog_data,
                                                       num_simulations=num_simulations, seed=seed)

    # populate result
    result.test_distribution = simulated_ll
    result.name = 'CL-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


def csep1_number_test(gridded_forecast, observed_catalog):
    """
    @asim
    Computes Number (N) test for Observed and Forecasts. Both data sets are expected to be in terms of event counts.

    We find the Total number of events in Observed Catalog and Forecasted Catalogs. Which are then employed to compute the probablities of
       (i) At least no. of events (delta 1)
       (ii) At most no. of events (delta 2) assuming the possionian distribution.

    Args:
        observation: Observed (Grided) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
        forecast:   Forecast of a Model (Grided) (Numpy Array)
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
    delta1 = 1.0 - scipy.stats.poisson.cdf(obs_cnt - epsilon, fore_cnt)
    delta2 = scipy.stats.poisson.cdf(obs_cnt + epsilon, fore_cnt)

    # store results
    result.test_distribution = 'poisson'
    result.name = 'N-Test'
    result.observed_statistic = obs_cnt
    result.quantile = (delta1, delta2)
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)
    result.fore_cnt = fore_cnt

    return result