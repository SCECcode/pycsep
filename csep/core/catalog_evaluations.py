from csep.utils.stats import get_quantiles
from csep.models import CatalogNumberTestResult

def number_test(forecast, observed_catalog):
    """ Performs the number test on a forecast specified as a collection of synthetic earthquake catalogs.

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

def spatial_test(forecast, catalog):
    """ Performs spatial test for data forecasts """
    raise NotImplementedError('spatial_test not implemented!')

def magnitude_test(forecast, catalog):
    """ Performs magnitude test for data forecasts """
    raise NotImplementedError('magnitude_test not implemented!')

def pseudolikelihood_test(forecast, catalog):
    """ Performas the pseudolikelihood test for data forecasts """
    raise NotImplementedError('pseudolikelihood_test not implemented!')

def calibration_test(evaluation_results):
    """ Performs the calibration test using multiple results """
    raise NotImplementedError('calibration_test not implemented!')

