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

    # call the primative version operating on ndarray
    out = csep1_t_test_ndarray(gridded_forecast1.data, gridded_forecast2.data,
                                                                                observed_catalog.event_count, alpha=alpha)

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
    see Rhoades, D. A., Schorlemmer, D., Gerstenberger, M. C., Christophersen, A., Zechar, J. D., & Imoto, M. (2011).
    Efficient testing of earthquake forecasting models. Acta Geophysica, 59(4), 728-747. doi:10.2478/s11600-011-0013-5

    Args:
        gridded_forecast_1 (numpy.ndarray): nd-array storing gridded rates
        gridded_forecast_2 (numpy.ndarray): nd-array storing gridded rates
        n_obs (float, int, numpy.ndarray): number of observed earthquakes, should be whole number and >= zero.
        alpha (float): tolerance level for the type-i error rate of the statistical test\

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
    The Wilcoxon One Sample Signed-Rank tes is the non parametric version of the one sample t test. 
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
    
    d = x - m

    d = numpy.compress(numpy.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        numpy.warnings.warn("Sample size too small for normal approximation.")

    r = scipy.stats.rankdata(abs(d))
    r_plus = numpy.sum((d > 0) * r, axis=0)
    r_minus = numpy.sum((d < 0) * r, axis=0)

    #For "two-sided", choose minimum of both
    T = min(r_plus, r_minus)
   
    #Correction to be intorduced 
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    replist, repnum = scipy.stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = numpy.sqrt(se / 24)

    # compute statistic and p-value using normal approximation
    #  z = (T - mn - d) / se Continuity correction. But We are not considering continuity correction.
    z = (T - mn) / se
    
    # 2, is multiplied for "two-sided" distribution
    prob = 2. * scipy.stats.distributions.norm.sf(abs(z))

    #Accept the NULL Hypothesis [Median(Xi-Yi) = Given value]. If probability is greater than 0.05
    #If Probability is smaller than 0.05, Reject the NULL Hypothesis, that Median(Xi-Yi) != Given Value
    w_test_eval = { 'z_statistic' : z,
                    'probability': prob }

    return w_test_eval

def csep1_w_test(gridded_forecast1, gridded_forecast2, observation, model_1='None', model_2='None'):
    
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
    N = numpy.sum(observation)      #Sum of all the observed earthquakes
    N1 = gridded_forecast1.event_count    #Total number of Forecasted earthquakes by Model 1
    N2 = gridded_forecast2.event_count     #Total number of Forecasted earthquakes by Model 2
    X1 = numpy.log(gridded_forecast1.data)      #Log of every element of Forecast 1
    X2 = numpy.log(gridded_forecast2.data)      #Log of every element of Forecast 2
    
    median_value = (N1 - N2) / N
    
    Diff = X1 - X2
    
    #W_test is One Sample Wilcoxon Signed Rank Test. It accepts the data only in 1D array. 
    x = Diff.ravel()  #Converting 2D Difference to 1D
    
    w_test_dic = csep1_w_test_ndarray(x, median_value)
    
    w_test_dic['model_name_1'] =  model_1
    w_test_dic['model_name_2'] =  model_2
    
    return w_test_dic

