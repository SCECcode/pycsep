#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:06:58 2020


@author: khawaja
"""
import numpy
import scipy.stats

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
                  ,'model_name': model_name} 
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

    
def t_test(observation, forecast_1, forecast_2, model_name_1='None', model_name_2='None'):
    
    """
    Computes T test statistic by comparing Forecast 1 and Forecast 2 Catalogs
    
    We compare Forecasat of Model 1 and with Forecast of Model 2. Information Gain is computed, which is then employed to compute T statistic.
    Confidence interval of Information Gain can be computed using T_critical
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be observed seismicity in each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    forecast_1: Forecast of a model_1 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    forecast_2: Forecast of model_2 (Grided) (Numpy Array) 
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    model_1:    The name of Model 1, with whom the comparision will be drawn
    model_2:    The name of second model, which will be compared with Model 1
    
    Returns
    A dictionary of following elements
                 { 'model_name_1': Name of Model 1,
                   'model_name_2': Name of Model 2,
                   't_statistic' : T statistic computed between forecast 1 and forecast 2,
                   't_critical': Critical value of T at 95% CDF, assuming 1-Tail distribution,
                   'information_gain': Information gain per earthquake of Model A over Model B,
                   'IG_lower':Lower bound pf Information Gain Confidence Interval,
                   'IG_upper': Upper bound of Information Gain Confidence Interval}
    """
    
    #Some Pre Calculations -  Because they are being used repeatedly. 
    N = numpy.sum(observation)      #Total number of observed earthquakes
    nbins = numpy.size(observation) #No. of Bins
    N1 = numpy.sum(forecast_1)      #Total number of Forecasted earthquakes by Model 1
    N2 = numpy.sum(forecast_2)      #Total number of Forecasted earthquakes by Model 2
    X1 = numpy.log(forecast_1)      #Log of every element of Forecast 1
    X2 = numpy.log(forecast_2)      #Log of every element of Forecast 2
    
    #Information Gain, using Equation (17)  of Rhoades et al. 2011
    Information_Gain = (numpy.sum(X1 - X2) - (N1-N2)) / N   #I beleieve, N is Total no. of Earthquakes here. 
                                
      
        
    #Compute variance of (X1-X2) using Equation (18)  of Rhoades et al. 2011
    first_term =  (numpy.sum(numpy.power((X1 - X2), 2))) / (nbins-1)    #I believe N is No. of Bins (nbins)  here?? So replacing N by bins
    second_term =   numpy.power(numpy.sum(X1 - X2), 2) / (numpy.power(nbins,2) - nbins)   #nbins in place of N
    forecast_variance = first_term - second_term 
     
    forecast_std = numpy.sqrt(forecast_variance)  
    t_statistic = Information_Gain / (forecast_std/ numpy.sqrt(nbins))

    #Obtaining the Criticial Value of T from T distribution.  
    df = nbins-1
    t_critical = scipy.stats.t.ppf(1-(0.05/2), df)   #Assuming 2-Tail Distribution  for 2 tail, divide 0.05/2.

    #Computing Information Gain Interval. 
    IG_lower = Information_Gain - (t_critical*forecast_std / numpy.sqrt(nbins))  #Assuming N = nbins here.
    IG_upper = Information_Gain + (t_critical*forecast_std / numpy.sqrt(nbins))  #Assuming N = nbins here.
    
    #If T value greater than T critical, Then both Lower and Upper Confidence Interval limits will be greater than Zero. 
    # If above Happens, Then It means that Forecasting Model 1 is better than Forecasting Model 2. 
    t_test_eval = { 'model_name_1': model_name_1,
                    'model_name_2': model_name_2,
                    't_critical' : t_critical,
                    't_statistic': t_statistic,
                    'information_gain': Information_Gain,
                    'IG_lower': IG_lower,
                    'IG_upper': IG_upper  }    
   
    return t_test_eval


def multiple_t_tests(observation, dic_forecast_1, list_dic_forecast_2):
    
    """ 
    We compare Forecast of Model 1 and with all the Forecasts given in "dic_forecast_2. 
    It considers Forecast 1 as a benchmark and computes T statistics and Informatuion Gain to see which is better. 
    Information Gain is computed, which is then employed to compute T statistic.
    Confidence interval of Information Gain can be computed using T_critical
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be observed seismicity in each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    dic_forecast_1: A dictionary of Forecast of a Model_1 consiisting of 2 elements
                    {'model_name': Name of Model with which comparisions is to be drawn
                     'forecast': Forecast of a Model with which comparisions is to be drawn  }
             
    list_forecast_2: A list of dictionaries of Forecasts, which are to be compared with forecast of Model 1.
                     Every dictionary in the list consists of 2 elements
                     {'model_name': Name of Model which is to be compared with Model 1
                     'Forecast': Forecast of a Model which is to be comparisioned forecast of Model 1  }
                
    Returns
    A List of dictionaries:
    Every dictionary in the list of dictionaries consists of following elecments
                {'model_name_1': Name of Mdoel 1 with which rest of the forecasts are compared,
                 'model_name_2': Name of Model 2, which is compared with forecast of Model 1,
                 't_critical': Critical value of T at 95% CDF, assuming 1-Tail distribution,
                 'information_gain': Information gain per earthquake of Model A over Model B,
                 'IG_lower':Lower bound pf Information Gain Confidence Interval,
                 'IG_upper': Upper bound of Information Gain Confidence Interval }
    """
    
    model_name_1 = dic_forecast_1["model_name"]
    forecast_1 = dic_forecast_1["forecast"]
    
    model_name_2 = []
    forecast_2 = []
    
    output_dic_list = []
    
    for i in range(numpy.size(list_dic_forecast_2)):
        model_name_2 = list_dic_forecast_2[i]["model_name"]
        forecast_2 = list_dic_forecast_2[i]["forecast"]
        
        out = t_test(observation, forecast_1, forecast_2, model_name_1, model_name_2)
        output_dic_list.append(out)
        
    return output_dic_list


    
def w_test_single_sample(x, m=0):
    
    """
    Calculate the Single Sample Wilcoxon signed-rank test for any sample. .
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
    z = (T - mn - d) / se #Continuty correction. But We are not considering continuty correction. 
    z = (T - mn) / se
    
    # 2, is multiplied for "two-sided" distribution
    prob = 2. * scipy.stats.distributions.norm.sf(abs(z))

    #Accept the NULL Hypothesis [Median(Xi-Yi) = Given value]. If probability is greater than 0.05
    #If Probability is smaller than 0.05, Reject the NULL Hypothesis, that Median(Xi-Yi) != Given Value
    w_test_eval = { 'z_statistic' : z,
                    'probability': prob }    
    return w_test_eval

def w_test(observation, forecast_1, forecast_2, model_1='None', model_2='None'):
    
    """
    Calculate the Single Sample Wilcoxon signed-rank test for "log(forecast_1(i))-log(forecast_2(i))".
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
    forecast_1: Forecast of a model_1 (Grided) (Numpy Array)
                A forecast has to be in terms of Average Number of Events in Each Bin 
                It can be anything greater than zero
    forecast_2: Forecast of model_2 (Grided) (Numpy Array) 
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
    nbins = numpy.size(observation) #No. of Bins
    N1 = numpy.sum(forecast_1)      #Total number of Forecasted earthquakes by Model 1
    N2 = numpy.sum(forecast_2)      #Total number of Forecasted earthquakes by Model 2
    X1 = numpy.log(forecast_1)      #Log of every element of Forecast 1
    X2 = numpy.log(forecast_2)      #Log of every element of Forecast 2
    
    median_value = (N1-N2)/nbins
    
    Diff = X1 - X2
    
    #W_test is One Sample Wilcoxon Signed Rank Test. It accepts the data only in 1D array. 
    x = Diff.flatten()  #Converting 2D Difference to 1D
    
    w_test_dic = w_test_single_sample(x, median_value)
    
    w_test_dic['model_name_1'] =  model_1
    w_test_dic['model_name_2'] =  model_2
    
    return w_test_dic



def multiple_w_tests(observation, dic_forecast_1, list_dic_forecast_2):
    
    """ 
    We compare Forecast of Model 1 and with all the Forecasts given in "list_dic_forecast_2" using  test. 
    It considers Forecast 1 as a benchmark and computes w test.
    
    Args
    observation:Observed (Grided) seismicity (Numpy Array):
                An Observation has to be observed seismicity in each Bin
                It has to be a either zero or positive integer only (No Floating Point)
    dic_forecast_1: A dictionary of Forecast of a Model_1 consiisting of 2 elements
                    {'model_name': Name of Model with which comparisions is to be drawn
                     'forecast': Forecast of a Model with which comparisions is to be drawn  }
             
    list_forecast_2: A list of dictionaries of Forecasts, which are to be compared with forecast of Model 1.
                     Every dictionary in the list consists of 2 elements
                     {'model_name': Name of Model which is to be compared with Model 1
                     'Forecast': Forecast of a Model which is to be comparisioned forecast of Model 1  }
                
    Returns
    A List of dictionaries:
    Every dictionary in the list of dictionaries consists of following elecments
                A dictionary of following elements
                {'model_name_1': Name of Model 1,
                 'model_name_2': Name of Model 2,
                 'z_statistic' : Z statistic computed between forecast 1 and forecast 2,
                 'probablity': Probablity value}
    """
    
    model_name_1 = dic_forecast_1["model_name"]
    forecast_1 = dic_forecast_1["forecast"]
    
    model_name_2 = []
    forecast_2 = []
    
    output_dic_list = []
    
    for i in range(numpy.size(list_dic_forecast_2)):
        model_name_2 = list_dic_forecast_2[i]["model_name"]
        forecast_2 = list_dic_forecast_2[i]["forecast"]
        
        out = w_test(observation, forecast_1, forecast_2, model_name_1, model_name_2)
        output_dic_list.append(out)
        
    return output_dic_list
    
    
