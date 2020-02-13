#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:06:58 2020


@author: khawaja
"""
import numpy
import scipy.stats

def poisson_log_likelihood(observation, forecast):      #This calls Poisson PMF function
    """Wrapper around scipy to compute the Poisson log-likelihood
    
    Args:
        observation: Observed (Grided) seismicity
        forecast: Forecast of a Model (Grided)
    
    Returns:
        Log-Liklihood values of between binned-observations and binned-forecasts
    """
    return numpy.log(scipy.stats.poisson.pmf(observation, forecast))


def _forecast_realization_inverse_cdf(random_matrix, forecast):
    """Wrapper around scipy inverse poisson cdf to compute new forecast using 
        actual forecast and random numbers between 0 and 1
    
    Args:
        random_matrix: Matrix of dimenions equal to forecast, containing random
                       numbers between 0 and 1.
        forecast: Actual Forecast of a Model (Grided)
    
    Returns:
        A new forecast which is the realization of Actual forecast
    """
        
    return scipy.stats.poisson.ppf(random_matrix, forecast) 


def l_test(observation, forecast, num_simulations=1000, model_name='None'):
    """
    Computes L test for Observed and Forecasted Catalogs
    
    We find the Joint Log Likelihood between Observed Catalog and Forecasted 
    Catalogs. Later on, "num_simulations" Forecast Realizations are generated using Forecasted Catalog and Random numbers through inverse poission CDF.
    Then Joint Likelihood of Forecast and Forecast Realizations are computed. 
    Actual Joint Log Likelihood and Simulated Joint Log Likelihoods are then employed to compute Quantile Score
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    num_simulation: Number of simulated Catalogs to be generated (Non-negative Integer)
                A non-negative integer indicating the number of realized forecasts to be generated
    model_name: The chosen name of model.             
    
    Returns
    A dictionary of 4 elements
                {'quantile' : Quantile Score of the Model,
                    'll_actual': joint Log Likelihood of Observation and Actual Forecast,
                     'll_simulation': Numpy array of Joint Likelihood of Simulated Forecasts 
                      'model_name': The name of the model, provided in the input}
    """
                                                #, seed=None
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
        

    ll_actual = numpy.sum(poisson_log_likelihood(observation, forecast))   

    ll_simulations = numpy.zeros(num_simulations)
    
    # compute simulated log-likelihoods
    for count in range(num_simulations):
        # random percentile for each independent bin
        random_matrix = numpy.random.random(forecast.shape) 
        # computes the inverse cdf of the poisson random variable
        forecast_realization = _forecast_realization_inverse_cdf(random_matrix, forecast)  
        ll_simulations[count] = numpy.sum(poisson_log_likelihood(forecast_realization, forecast))
    quantile_score = numpy.sum(ll_simulations <= ll_actual) / num_simulations #Apply 3rd UNIT TEST here on Quantile Score. 
    

    L_Test_Eval = { 'model_name': model_name,
                    'quantile' : quantile_score,
                    'll_actual': ll_actual,
                     'll_simulation':ll_simulations}
     
    return L_Test_Eval
        
def _m_test_prep(observation, forecast):
    """
    This function prepares observation and forecasts by summing up all the values accross Spance-bins (x-axis, in our case)
    After summing up on Space-bins, it returns a 1-D vector containing only magnitude bins
    After summation, it normalizes the forecasts are normalized by factor of (N_obs/_forecasts)
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    
    Returns
    observations_magbins: Observed seismicity (Grided only accross Magnitude bins) 
    norm_forecast_magbins: Normalized Forecasted seismicity (Grided only accross Magnitude bins) 
    """
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
        
    N_obs = numpy.sum(observation)
    N_fcst = numpy.sum(forecast)
    
    observations_magbins = numpy.sum(observation, axis=0)
    forecast_magbins = numpy.sum(forecast,axis=0)
    
    norm_forecast_magbins = forecast_magbins*N_obs/N_fcst
    
    return observations_magbins, norm_forecast_magbins



def m_test(observation, forecast, num_simulations = 1000, model_name = 'None'):
    """
    Computes M-test for Observed and Forecasted Catalogs
    -It must be noted that observation and forecasts are arranged in following form
                    [n-space bins, n-mag bins]
    Therefore for M-Test all the values in n-space bins are summed, and only n-mag bins are left
    We find the Joint Log Likelihood between Observed Catalog and Forecasted 
    Catalogs. Later on, "num_simulations" Forecast Realizations are generated using Forecasted Catalog and Random numbers through inverse poission CDF.
    Then Joint Likelihood of Forecast and Forecast Realizations are computed. 
    Actual Joint Log Likelihood and Simulated Joint Log Likelihoods are then employed to compute Quantile Score
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    num_simulation: Number of simulated Catalogs to be generated (Non-negative Integer)
                A non-negative integer indicating the number of realized forecasts to be generated
    model_name: A chosen name of the model is also required.
    Returns
    A dictionary of 4 elements
                {'quantile' : Quantile Score of the Model,
                    'll_actual': joint Log Likelihood of Observation and Actual Forecast,
                     'll_simulation': Numpy array of Joint Likelihood of Simulated Forecasts 
                     'model_name': The name of the model, provided in the input}
    """
                                                #, seed=None
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
    
    [observations_magbins, norm_forecast_magbins] = _m_test_prep(observation,forecast)
  
    M_Test_Eval = l_test(observations_magbins, norm_forecast_magbins, num_simulations, model_name)
    return M_Test_Eval


def _s_test_prep(observation, forecast):
    """
    This function prepares observation and forecasts by summing up all the values accross Magnitude-bins (y-axis, in our case)
    After summing up on Magnitude-bins, it returns a 1-D vector containing only Space bins
    After summation, it normalizes the forecasts are normalized by factor of (N_obs/_forecasts)
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    
    Returns
    observations_magbins: Observed seismicity (Grided only accross Space bins) 
    norm_forecast_magbins: Normalized Forecasted seismicity (Grided only accross Space bins) 
    """
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
        
    N_obs = numpy.sum(observation)
    N_fcst = numpy.sum(forecast)
    
    observations_spacebins = numpy.sum(observation, axis=1)
    forecast_spacebins = numpy.sum(forecast,axis=1)
    
    norm_forecast_spacebins = forecast_spacebins*N_obs/N_fcst
    
    return observations_spacebins, norm_forecast_spacebins



def s_test(observation, forecast, num_simulations = 1000, model_name = 'None'):
    """
    Computes M-test for Observed and Forecasted Catalogs
    -It must be noted that observation and forecasts are arranged in following form
                    [n-space bins, n-mag bins]
    Therefore for M-Test all the values in n-space bins are summed, and only n-mag bins are left
    We find the Joint Log Likelihood between Observed Catalog and Forecasted 
    Catalogs. Later on, "num_simulations" Forecast Realizations are generated using Forecasted Catalog and Random numbers through inverse poission CDF.
    Then Joint Likelihood of Forecast and Forecast Realizations are computed. 
    Actual Joint Log Likelihood and Simulated Joint Log Likelihoods are then employed to compute Quantile Score
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    num_simulation: Number of simulated Catalogs to be generated (Non-negative Integer)
                A non-negative integer indicating the number of realized forecasts to be generated
    model_name: A chosen name of model is required
    Returns
    A dictionary of 4 elements
                {'quantile' : Quantile Score of the Model,
                    'll_actual': joint Log Likelihood of Observation and Actual Forecast,
                     'll_simulation': Numpy array of Joint Likelihood of Simulated Forecasts 
                      'model_name': The name of the model, provided in the input}
    """
                                                #, seed=None
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
    
    [observations_spacebins, norm_forecast_spacebins] = _s_test_prep(observation,forecast)
  
    S_Test_Eval = l_test(observations_spacebins, norm_forecast_spacebins, num_simulations, model_name)

    return S_Test_Eval

def n_test(observation, forecast, model_name = 'None'):
    """
    Computes Number (N) test for Observed and Forecasted Catalogs
    
    We find the Total number of events in Observed Catalog and Forecasted 
    Catalogs. Which are then employed to compute the probablities of 
    (i) At least no. of events (delta 1)
    (ii) At most no. of events (delta 2)
    assuming the possionian distribution.
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    model_name: A chosen name of model is required
    Returns
    A dictionary of 3 elements
                {'delta1' : delta1,
                  'delta2': delta2,
                  'model_name': The name of the model, provided in the input}
    """

    Epsilon = 1e-6
     
    delta1 = 1.0-scipy.stats.poisson.cdf(numpy.sum(observation) - Epsilon, numpy.sum(forecast))
    
    delta2 = scipy.stats.poisson.cdf(numpy.sum(observation) + Epsilon, numpy.sum(forecast))
    
    Number_Test_Eval = {'delta1' : delta1,
                  'delta2': delta2
                  ,'name': model_name} 
    return Number_Test_Eval


def _conditional_l_test_prep(observation, forecast):
    """
    This function prepares the forecasts by normalizing them by a factor of (N_obs/_forecasts)
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    
    Returns
    norm_forecast: Normalized Forecasted seismicity (Grided only accross Magnitude bins) 
    """
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
        
    N_obs = numpy.sum(observation)
    N_fcst = numpy.sum(forecast)
    

    norm_forecast = forecast*N_obs/N_fcst
    
    return norm_forecast



def conditional_l_test(observation, forecast, num_simulations = 1000, model_name = 'None'):
    """
    Computes Conditional L-test for Observed and Forecasted Catalogs
    This function first normalizes forecasts by a factor of (N_obs/_forecasts)
    
    We find the Joint Log Likelihood between Observed Catalog and Normalized Forecast
    Catalogs. Later on, "num_simulations" Normalized Forecast Realizations are generated using Forecasted Catalog and Random numbers through inverse poission CDF.
    Then Joint Likelihood of Forecast and Forecast Realizations are computed. 
    Actual Joint Log Likelihood and Simulated Joint Log Likelihoods are then employed to compute Quantile Score
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be Number of Events in Each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast:   Forecast of a Model (Grided) (Numpy Arrazy)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    num_simulation: Number of simulated Catalogs to be generated (Non-negative Integer)
                A non-negative integer indicating the number of realized forecasts to be generated
    model_name: A chosen name of the model is also required.
    Returns
    A dictionary of 4 elements
                {'quantile' : Quantile Score of the Model,
                    'll_actual': joint Log Likelihood of Observation and Actual Forecast,
                     'll_simulation': Numpy array of Joint Likelihood of Simulated Forecasts 
                     'model_name': The name of the model, provided in the input}
    """
    
    if not isinstance(observation, numpy.ndarray):
        raise TypeError('observation must be numpy.ndarray')
    if not isinstance(forecast, numpy.ndarray):
        raise TypeError('forecast must be numpy.ndarray')
    
    norm_forecast = _conditional_l_test_prep(observation,forecast)
    
    Conditional_L_Test_Eval = l_test(observation, norm_forecast, num_simulations, model_name)
    return Conditional_L_Test_Eval
