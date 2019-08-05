from collections import namedtuple

import numpy
import tqdm
from numba import jit

import time
from csep.utils.stats import cumulative_square_dist
from csep.utils.constants import CSEP_MW_BINS
from csep.utils import flat_map_to_ndarray

# implementing plotting routines as functions
from stats import _get_quantiles

EvaluationResult = namedtuple('EvaluationResult', ['test_distribution',
                                                   'name',
                                                   'observed_statistic',
                                                   'quantile',
                                                   'status',
                                                   'sim_catalog_repr',
                                                   'obs_catalog_repr',
                                                   'sim_name',
                                                   'obs_name'])

def number_test(stochastic_event_sets, observation, event_counts=None):
    # get number of events for observations and simulations
    if not event_counts:
        sim_counts = []
        for catalog in stochastic_event_sets:
            sim_counts.append(catalog.event_count)
    else:
        sim_counts = event_counts
    observation_count = observation.event_count
    # get delta_1 and delta_2 values
    delta_1, delta_2 = _get_quantiles(sim_counts, observation_count)
    # prepare result
    result = EvaluationResult(test_distribution=sim_counts,
                              name='N-Test',
                              observed_statistic=observation_count,
                              quantile=(delta_1, delta_2),
                              status='Normal',
                              sim_catalog_repr=str(stochastic_event_sets[0]),
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result

def pseudo_likelihood_test(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    """
    This test makes a discrete approximation of the continuous conditional rate-density function
    that governs the forecast. The intensity is magnitude independent, so only varies spatially and not
    with magnitude. This provides a scalar conditional intensity value in each cell.

    We build the distribution of the test statistic under the null hypothesis by computing the psuedo-likelihood score for
    each catalog given the approximate conditional intensity function, by summing likelihoods over each event in the set.
    We normalize this by the expected number of events from the conditional intensity function.

    Note: Some of these statements are vectorized for performance.

    If writing a custom reduce_func, this needs to reduce of axis=1 of a numpy.ndarray. For example, c = numpy.mean(a, axis=1).

    Args:
        stochastic_event_sets: list of catalogs
        observation: catalog
        apprx_log_rate_density: 1d array corresponding to region. density should be counts / dh / dh / time_interval (yr)
        normed: this normalizes by the cell event-counts to isolate the spatial distribution, effectively the s-test

    Returns:
        p-value: float
    """

    # integrating, assuming that all cats in ses have same region
    region = stochastic_event_sets[0].region
    expected_cond_count = numpy.sum(apprx_rate_density) * region.dh * region.dh * time_interval
    gridded_obs = observation.gridded_event_counts()
    name = 'L-Test'

    # build likelihood distribution from ses
    test_distribution = []
    t0 = time.time()
    for i, catalog in enumerate(stochastic_event_sets):
        gridded_cat = catalog.gridded_event_counts()
         # compute likelihood for each event, ignoring areas with 0 expectation
        gridded_cat_ma = numpy.ma.masked_where(gridded_cat == 0, gridded_cat)
        apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_cat_ma.mask)
        likelihood = numpy.ma.sum(gridded_cat_ma * numpy.ma.log10(apprx_rate_density_ma)) - expected_cond_count
        test_distribution.append(likelihood)
        # report real-time stats
        if (i+1) % 5000 == 0:
            t1 = time.time()
            print(f"Processed {i+1} catalogs in {t1-t0} seconds.")

    # compute psuedo-likelihood for comcat
    gridded_obs_ma = numpy.ma.masked_where(gridded_obs == 0, gridded_obs)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_obs_ma.mask)
    comcat_likelihood = numpy.ma.sum(gridded_obs_ma * numpy.ma.log10(apprx_rate_density_ma)) - expected_cond_count

    # determine outcome of evaluation, check for infinity
    _, quantile = _get_quantiles(test_distribution, comcat_likelihood)

    # Signals outcome of test
    message = "Normal"

    # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
    # either normal and wrong or udetermined (undersampled)
    if numpy.isclose(quantile, 0.0) or numpy.isclose(quantile, 1.0):

        # undetermined failure of the test
        if numpy.isinf(comcat_likelihood):

            # Build message
            message = f"undetermined. Infinite likelihood scores found."

    # build evaluation result
    result = EvaluationResult(test_distribution=test_distribution,
                              name=name,
                              observed_statistic=comcat_likelihood,
                              quantile=quantile,
                              status=message,
                              sim_catalog_repr=str(stochastic_event_sets[0]),
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def spatial_test(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    """
    Computes an evaluation of a forecast spatially by isolating the spatial distribution of the approximate conditional
    rate density. It does this by normalizing the expected conditional rate density by the ratio between observed and
    simulated event counts. This effectively removes the rate component from the forecast.

    Args:
        stochastic_event_sets: list of catalogs
        observation: catalog
        apprx_rate_density: 1d array corresponding to region,

    Returns:
        quantile: p(X ≤ x) from the test distribution and observation

    """
    # integrating, assuming that all cats in ses have same region
    name = 'S-Test'

    # build likelihood distribution from ses
    region = stochastic_event_sets[0].region
    test_distribution = []
    # this could be io based if iterator is passed
    for catalog in tqdm.tqdm(stochastic_event_sets, total=len(stochastic_event_sets)):
        gridded_rate_cat = catalog.gridded_event_counts() / time_interval
        # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio
        normalizing_factor = observation.event_count / catalog.event_count
        gridded_rate_cat_norm = normalizing_factor * gridded_rate_cat
        # compute likelihood for each event, ignoring areas with 0 expectation
        gridded_rate_cat_norm_ma = numpy.ma.masked_where(gridded_rate_cat_norm == 0, gridded_rate_cat_norm)
        apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_rate_cat_norm_ma.mask)
        likelihood = numpy.ma.sum(gridded_rate_cat_norm_ma * numpy.ma.log10(apprx_rate_density_ma))
        test_distribution.append(likelihood)

    # compute psuedo-likelihood for comcat
    gridded_obs_rate = observation.gridded_event_counts() / time_interval
    gridded_obs_rate_ma = numpy.ma.masked_where(gridded_obs_rate == 0, gridded_obs_rate)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_obs_rate_ma.mask)
    comcat_likelihood = numpy.ma.sum(gridded_obs_rate_ma * numpy.ma.log10(apprx_rate_density_ma))

    # determine outcome of evaluation, check for infinity
    _, quantile = _get_quantiles(test_distribution, comcat_likelihood)

    # Signals outcome of test
    message = "Normal"
    # Deal with case with cond. rate. density func has zeros
    if numpy.isclose(quantile, 0.0) or numpy.isclose(quantile, 1.0):
        # undetermined failure of the test
        if numpy.isinf(comcat_likelihood):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."
    # build evaluation result
    result = EvaluationResult(test_distribution=test_distribution,
                              name=name,
                              observed_statistic=comcat_likelihood,
                              quantile=quantile,
                              status=message,
                              sim_catalog_repr=str(stochastic_event_sets[0]),
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result

def magnitude_test(stochastic_event_sets, observation, mag_bins=CSEP_MW_BINS):
    """
    Compares the observed magnitude distribution with the forecasted magnitude distribution. It does this by generating a
    union distribution (distribution of magnitudes for all events in all stochastic event sets). Then we can compare the
    magnitude distribution in each individual stochastic event set to build the test distribution under the null hypothesis.
    Like the other tests, we report the p-value of the observation given the distribution of the test statistics under the
    null hypothesis.

    Right now this evaluations scales by the ratio of the event counts in the stochastic event sets and the observed catalog.
    We randomly sample N_Obs_Events to ensure the variance of the test distribution is consistent with that of the observations.


    Args:
        stochastic_event_set: list of catalogs
        observation: observation catalog

    Returns:
        quantile: quantile score P(X ≤ x)
    """
    # get data that we need
    stochastic_event_sets_magnitudes = [cat.get_magnitudes() for cat in stochastic_event_sets]
    union_catalog = flat_map_to_ndarray(stochastic_event_sets_magnitudes)
    n_union_events = len(union_catalog)

    # build bins to compute histograms from experiment details

    obs_magnitudes = observation.get_magnitudes()
    n_obs_events = len(obs_magnitudes)

    # normalize by event counts
    scale = n_obs_events / n_union_events
    union_histogram = numpy.histogram(union_catalog, bins=mag_bins)[0] * scale

    # compute histograms and convert to rates
    obs_histogram, bin_edges = numpy.histogram(obs_magnitudes, bins=mag_bins)

    # build test-distribution
    test_distribution = []
    for ses_mags in stochastic_event_sets_magnitudes:
        n_ses_mags = len(ses_mags)
        if n_ses_mags == 0:
            scale = 0
        else:
            scale = n_obs_events / n_ses_mags
        ses_histogram = numpy.histogram(ses_mags, bins=mag_bins)[0] * scale
        # this distribution might not have the expected variance given n_obs_events.
        d_statistic = cumulative_square_dist(ses_histogram, union_histogram)
        test_distribution.append(d_statistic)

    # compute statistic from the observation
    obs_d_statistic = cumulative_square_dist(obs_histogram, union_histogram)

    # score evaluation
    _, quantile = _get_quantiles(test_distribution, obs_d_statistic)

    result = EvaluationResult(test_distribution=test_distribution,
                              name='M-Test',
                              observed_statistic=obs_d_statistic,
                              quantile=quantile,
                              status='Normal',
                              sim_catalog_repr=str(stochastic_event_sets[0]),
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def _compute_likelihood(gridded_data, apprx_rate_density, expected_cond_count, time_interval, n_obs, n_gridded):
    gridded_cat_ma = numpy.ma.masked_where(gridded_data == 0, gridded_data)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_cat_ma.mask)
    likelihood = numpy.ma.sum(gridded_cat_ma * numpy.ma.log10(apprx_rate_density_ma)) - expected_cond_count
    # compute spatial 'likelihood'
    gridded_rate_cat = gridded_data / time_interval
    # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio
    normalizing_factor = n_obs / n_gridded
    gridded_rate_cat_norm = normalizing_factor * gridded_rate_cat
    # compute likelihood for each event, ignoring areas with 0 expectation
    gridded_rate_cat_norm_ma = numpy.ma.masked_where(gridded_rate_cat_norm == 0, gridded_rate_cat_norm)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_rate_cat_norm_ma.mask)
    likelihood_norm = numpy.ma.sum(gridded_rate_cat_norm_ma * numpy.ma.log10(apprx_rate_density_ma))
    return(likelihood, likelihood_norm)

def combined_likelihood_and_spatial(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    # integrating, assuming that all cats in ses have same region
    region = stochastic_event_sets[0].region
    n_cat = len(stochastic_event_sets)
    expected_cond_count = numpy.sum(apprx_rate_density) * region.dh * region.dh * time_interval
    gridded_obs = observation.gridded_event_counts()
    n_obs = observation.get_number_of_events()

    # build likelihood distribution from ses
    test_distribution_likelihood = numpy.empty(n_cat)
    test_distribution_spatial = numpy.empty(n_cat)
    t0 = time.time()
    for i, catalog in enumerate(stochastic_event_sets):
        gridded_cat = catalog.gridded_event_counts()
        n_ses = catalog.get_number_of_events()
        # compute likelihood for each event, ignoring areas with 0 expectation,
        lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density, expected_cond_count, time_interval, n_obs, n_obs)
        # store results
        test_distribution_likelihood[i] = lh
        test_distribution_spatial[i] = lh_norm
        if (i+1) % 5000 == 0:
            t1 = time.time()
            print(f"Processed {i+1} catalogs in {t1-t0} seconds.")

    obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density, expected_cond_count, time_interval, n_obs, n_ses)

    # determine outcome of evaluation, check for infinity
    _, quantile_likelihood = _get_quantiles(test_distribution_likelihood, obs_lh)
    _, quantile_spatial = _get_quantiles(test_distribution_spatial, obs_lh_norm)

    # Signals outcome of test
    message = "Normal"
    # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
    # either normal and wrong or udetermined (undersampled)
    if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
        # undetermined failure of the test
        if numpy.isinf(obs_lh):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."
    # build evaluation result
    result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood,
                                         name='L-Test',
                                         observed_statistic=obs_lh,
                                         quantile=quantile_likelihood,
                                         status=message,
                                         sim_catalog_repr=str(stochastic_event_sets[0]),
                                         obs_catalog_repr=str(observation),
                                         sim_name=stochastic_event_sets[0].name,
                                         obs_name=observation.name)
    # find out if there are issues with the test
    if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
        # undetermined failure of the test
        if numpy.isinf(obs_lh_norm):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."

    # build evaluation result
    result_spatial = EvaluationResult(test_distribution=test_distribution_spatial,
                                      name='S-Test',
                                      observed_statistic=obs_lh_norm,
                                      quantile=quantile_spatial,
                                      status=message,
                                      sim_catalog_repr=str(stochastic_event_sets[0]),
                                      obs_catalog_repr=str(observation),
                                      sim_name=stochastic_event_sets[0].name,
                                      obs_name=observation.name)
    return (result_likelihood, result_spatial)

def interevent_time_test(stochastic_event_sets, observation):
    """
    These compare the inter-event time distribution of the forecasts with the observation. It works similarly to
    the magnitude test. First, we build the union distribution

    Args:
        stochastic_event_sets:
        observation:

    Returns:

    """
    pass

def inter_event_distribution_test(stochastic_event_sets, observation):
    pass

def total_event_rate_distribution_test(stochastic_event_sets, observation):
    pass

def depth_distribution_test(stohastic_event_sets, observation):
    pass

def bvalue_distribution_test(stochastic_event_sets, observation):
    pass

