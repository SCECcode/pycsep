# Third-Party Imports
import numpy
import scipy.stats

# PyCSEP imports
from csep.core.exceptions import CSEPEvaluationException
from csep.models import (
    CatalogNumberTestResult,
    CatalogSpatialTestResult,
    CatalogMagnitudeTestResult,
    CatalogPseudolikelihoodTestResult,
    CalibrationTestResult
)
from csep.utils.calc import _compute_likelihood
from csep.utils.stats import get_quantiles, cumulative_square_diff


def number_test(forecast, observed_catalog):
    """ Performs the number test on a catalog-based forecast.

    The number test builds an empirical distribution of the event counts for each data. By default, this
    function does not perform any filtering on the catalogs in the forecast or observation. These should be handled
    outside of the function.

    Args:
        forecast (:class:`csep.core.forecasts.CatalogForecast`): forecast to evaluate
        observed_catalog (:class:`csep.core.catalogs.AbstractBaseCatalog`): evaluation data

    Returns:
        evaluation result (:class:`csep.models.EvaluationResult`): evaluation result
    """
    event_counts = []
    for catalog in forecast:
        event_counts.append(catalog.event_count)
    obs_count = observed_catalog.event_count
    delta_1, delta_2 = get_quantiles(event_counts, obs_count)
    # prepare result
    result = CatalogNumberTestResult(test_distribution=event_counts,
                                     name='Catalog N-Test',
                                     observed_statistic=obs_count,
                                     quantile=(delta_1, delta_2),
                                     status='Normal',
                                     obs_catalog_repr=str(observed_catalog),
                                     sim_name=forecast.name,
                                     min_mw=forecast.min_magnitude,
                                     obs_name=observed_catalog.name)
    return result

def spatial_test(forecast, observed_catalog):
    """ Performs spatial test for catalog-based forecasts.



        Args:
            forecast: CatalogForecast
            observed_catalog: CSEPCatalog filtered to be consistent with the forecast

        Returns:
            CatalogSpatialTestResult
    """

    if forecast.region is None:
        raise CSEPEvaluationException("Forecast must have region member to perform spatial test.")

    # get observed likelihood
    if observed_catalog.event_count == 0:
        print(f'Skipping spatial tests because no events in observed catalog.')
        return None

    test_distribution = []

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates()

    expected_cond_count = forecast.expected_rates.sum()
    forecast_mean_spatial_rates = forecast.expected_rates.spatial_counts()

    # summing over spatial counts ensures that the correct number of events are used; even through the catalogs should
    # be filtered before calling this function
    gridded_obs = observed_catalog.spatial_counts()
    n_obs = numpy.sum(gridded_obs)

    # iterate through catalogs in forecast and compute likelihood
    for catalog in forecast:
        gridded_cat = catalog.spatial_counts()
        _, lh_norm = _compute_likelihood(gridded_cat, forecast_mean_spatial_rates, expected_cond_count, n_obs)
        test_distribution.append(lh_norm)

    _, obs_lh_norm = _compute_likelihood(gridded_obs, forecast_mean_spatial_rates, expected_cond_count, n_obs)
    # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
    message = "normal"
    if obs_lh_norm == -numpy.inf:
        idx_good_sim = forecast_mean_spatial_rates != 0
        new_gridded_obs = gridded_obs[idx_good_sim]
        new_n_obs = numpy.sum(new_gridded_obs)
        print(f"Found -inf as the observed likelihood score. "
              f"Assuming event(s) occurred in undersampled region of forecast.\n"
              f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
        if new_n_obs == 0:
            print(
                f'Skipping pseudo-likelihood based tests for because no events in observed catalog '
                f'after correcting for under-sampling in forecast.'
            )
            return None

        new_ard = forecast_mean_spatial_rates[idx_good_sim]
        # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
        # statistic will not be computed correctly.
        _, obs_lh_norm = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count, n_obs)
        message = "undersampled"

    # check for nans here and remove from spatial distribution
    test_distribution_spatial_1d = numpy.array(test_distribution)
    if numpy.isnan(numpy.sum(test_distribution_spatial_1d)):
        test_distribution_spatial_1d = test_distribution_spatial_1d[~numpy.isnan(test_distribution_spatial_1d)]

    if n_obs == 0 or numpy.isnan(obs_lh_norm):
        message = "not-valid"
        delta_1, delta_2 = -1, -1
    else:
        delta_1, delta_2 = get_quantiles(test_distribution_spatial_1d, obs_lh_norm)

    result = CatalogSpatialTestResult(test_distribution=test_distribution_spatial_1d,
                                      name='S-Test',
                                      observed_statistic=obs_lh_norm,
                                      quantile=(delta_1, delta_2),
                                      status=message,
                                      min_mw=forecast.min_magnitude,
                                      obs_catalog_repr=str(observed_catalog),
                                      sim_name=forecast.name,
                                      obs_name=observed_catalog.name)

    return result


def magnitude_test(forecast, observed_catalog):
    """ Performs magnitude test for catalog-based forecasts """
    test_distribution = []

    if forecast.region.magnitudes is None:
        raise CSEPEvaluationException("Forecast must have region.magnitudes member to perform magnitude test.")

    # short-circuit if zero events
    if observed_catalog.event_count == 0:
        print("Cannot perform magnitude test when observed event count is zero.")
        return None

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates()

    # returns the average events in the magnitude bins
    union_histogram = forecast.expected_rates.magnitude_counts()
    n_union_events = numpy.sum(union_histogram)
    obs_histogram = observed_catalog.magnitude_counts()
    n_obs = numpy.sum(obs_histogram)
    union_scale = n_obs / n_union_events
    scaled_union_histogram = union_histogram * union_scale

    # compute the test statistic for each catalog
    for catalog in forecast:
        mag_counts = catalog.magnitude_counts()
        n_events = numpy.sum(mag_counts)
        if n_events == 0:
            # print("Skipping to next because catalog contained zero events.")
            continue
        scale = n_obs / n_events
        catalog_histogram = mag_counts * scale
        # compute magnitude test statistic for the catalog
        test_distribution.append(
            cumulative_square_diff(numpy.log10(catalog_histogram + 1), numpy.log10(scaled_union_histogram + 1))
        )

    # compute observed statistic
    obs_d_statistic = cumulative_square_diff(numpy.log10(obs_histogram + 1), numpy.log10(scaled_union_histogram + 1))

    # score evaluation
    delta_1, delta_2 = get_quantiles(test_distribution, obs_d_statistic)

    # prepare result
    result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                              name='M-Test',
                              observed_statistic=obs_d_statistic,
                              quantile=(delta_1, delta_2),
                              status='Normal',
                              min_mw=forecast.min_magnitude,
                              obs_catalog_repr=str(observed_catalog),
                              obs_name=observed_catalog.name,
                              sim_name=forecast.name)

    return result

def pseudolikelihood_test(forecast, observed_catalog):
    """ Performs the spatial pseudolikelihood test for catalog forecasts.

    Performs the spatial pseudolikelihood test as described by Savran et al., 2020. The tests uses a pseudolikelihood
    statistic computed from the expected rates in spatial cells. A pseudolikelihood test based on space-magnitude bins
    is in a development mode and does not exist currently.

    Args:
        forecast: :class:`csep.core.forecasts.CatalogForecast`
        observed_catalog: :class:`csep.core.catalogs.AbstractBaseCatalog`
    """

    if forecast.region is None:
        raise CSEPEvaluationException("Forecast must have region member to perform spatial test.")

    # get observed likelihood
    if observed_catalog.event_count == 0:
        print(f'Skipping pseudolikelihood test because no events in observed catalog.')
        return None

    test_distribution = []

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        _ = forecast.get_expected_rates()

    expected_cond_count = forecast.expected_rates.sum()
    forecast_mean_spatial_rates = forecast.expected_rates.spatial_counts()

    # summing over spatial counts ensures that the correct number of events are used; even through the catalogs should
    # be filtered before calling this function
    gridded_obs = observed_catalog.spatial_counts()
    n_obs = numpy.sum(gridded_obs)

    for catalog in forecast:
        gridded_cat = catalog.spatial_counts()
        plh, _ = _compute_likelihood(gridded_cat, forecast_mean_spatial_rates, expected_cond_count, n_obs)
        test_distribution.append(plh)

    obs_plh, _ = _compute_likelihood(gridded_obs, forecast_mean_spatial_rates, expected_cond_count, n_obs)
    # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
    message = "normal"
    if obs_plh == -numpy.inf:
        idx_good_sim = forecast_mean_spatial_rates != 0
        new_gridded_obs = gridded_obs[idx_good_sim]
        new_n_obs = numpy.sum(new_gridded_obs)
        print(f"Found -inf as the observed likelihood score. "
              f"Assuming event(s) occurred in undersampled region of forecast.\n"
              f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
        if new_n_obs == 0:
            print(
                f'Skipping pseudo-likelihood based tests for because no events in observed catalog '
                f'after correcting for under-sampling in forecast.'
            )
            return None

        new_ard = forecast_mean_spatial_rates[idx_good_sim]
        # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
        # statistic will not be computed correctly.
        obs_plh, _ = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count, n_obs)
        message = "undersampled"

    # check for nans here
    test_distribution_1d = numpy.array(test_distribution)
    if numpy.isnan(numpy.sum(test_distribution_1d)):
        test_distribution_1d = test_distribution_1d[~numpy.isnan(test_distribution_1d)]

    if n_obs == 0 or numpy.isnan(obs_plh):
        message = "not-valid"
        delta_1, delta_2 = -1, -1
    else:
        delta_1, delta_2 = get_quantiles(test_distribution_1d, obs_plh)

    # prepare evaluation result
    result = CatalogPseudolikelihoodTestResult(
        test_distribution=test_distribution_1d,
        name='PL-Test',
        observed_statistic=obs_plh,
        quantile=(delta_1, delta_2),
        status=message,
        min_mw=forecast.min_magnitude,
        obs_catalog_repr=str(observed_catalog),
        sim_name=forecast.name,
        obs_name=observed_catalog.name
    )

    return result

def calibration_test(evaluation_results, delta_1=False):
    """ Perform the calibration test by computing a Kilmogorov-Smirnov test of the observed quantiles against a uniform
    distribution.

        Args:
            evaluation_results: iterable of evaluation result objects
            delta_1 (bool): use delta_1 for quantiles. default false -> use delta_2 quantile score for calibration test
    """

    idx = 0 if delta_1 else 1
    quantiles = [result.quantile[idx] for result in evaluation_results]
    ks, p_value = scipy.stats.kstest(quantiles, 'uniform')

    result = CalibrationTestResult(
        test_distribution = quantiles,
        name=f'{evaluation_results[0].name} Calibration Test',
        observed_statistic=ks,
        quantile=p_value,
        status='normal',
        min_mw = evaluation_results[0].min_mw,
        obs_catalog_repr=evaluation_results[0].obs_catalog_repr,
        sim_name=evaluation_results[0].sim_name,
        obs_name=evaluation_results[0].obs_name
    )

    return result


