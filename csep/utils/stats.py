import numpy
import scipy.special
import scipy.stats


def sup_dist(cdf1, cdf2):
    """
    given two cumulative distribution functions, compute the supremum of the set of absolute distances.

    note:
        this function does not check that the ecdfs are ordered or balanced. beware!
    """
    return numpy.max(numpy.absolute(cdf2 - cdf1))

def sup_dist_na(data1, data2):
    """
    computes the ks statistic for two ecdfs that are not necessarily aligned on the same values. performs this
    operation by merging the two datasets together. this is taken from the 2sample ks test in the scipy codebase

    Args:
        data1: (numpy array like)
        data2: (numpy array like)

    Returns:
        ks: sup dist from the two cdf functions
    """
    data1, data2 = map(numpy.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = numpy.sort(data1)
    data2 = numpy.sort(data2)
    data_all = numpy.concatenate([data1,data2])
    cdf1 = numpy.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (numpy.searchsorted(data2,data_all,side='right'))/(1.0*n2)
    d = numpy.max(numpy.absolute(cdf1-cdf2))
    return d

def cumulative_square_diff(cdf1, cdf2):
    """
    given two cumulative distribution functions, compute the cumulative sq. diff of the set of distances.

    note:
        this function does not check that the ecdfs are ordered or balanced. beware!

    Args:
        cdf1: ndarray
        cdf2: ndarray

    Returns:
        cum_dist: scalar distance metric for the histograms

    """
    return numpy.sum((cdf2 - cdf1)**2)

def binned_ecdf(x, vals):
    """
    returns the statement P(X ≤ x) for val in vals.
    vals must be monotonically increasing and unqiue.

    returns:
        tuple: sorted vals, and ecdf computed at vals
    """
    # precompute ecdf for x: returns(sorted(x), ecdf())
    if len(x) == 0:
        return None
    ex, ey = ecdf(x)
    cdf = numpy.array(list(map(lambda val: less_equal_ecdf(x, val, cdf=(ex, ey)), vals)))
    return vals, cdf

def ecdf(x):
    """
    Compute the ecdf of vector x. This does not contain zero, should be equal to 1 in the last value
    to satisfy F(x) == P(X ≤ x).

    Args:
        x (numpy.array): vector of values

    Returns:
        xs (numpy.array), ys (numpy.array)
    """
    xs = numpy.sort(x)
    ys = numpy.arange(1, len(x) + 1) / float(len(x))
    return xs, ys

def greater_equal_ecdf(x, val, cdf=()):
    """
    Given val return P(x ≥ val).

    Args:
        x (numpy.array): set of values
        val (float): value
        ecdf (tuple): ecdf of x, should be tuple (sorted(x), ecdf(x))

    Returns:
        (float): probability that x ≤ val
    """
    x = numpy.asarray(x)
    if x.shape[0] == 0:
        return None
    if not cdf:
        ex, ey = ecdf(x)
    else:
        ex, ey = cdf

    eyc = ey[::-1]

    # some short-circuit cases for discrete distributions; x is sorted, but reversed.
    if val > ex[-1]:
        return 0.0
    if val < ex[0]:
        return 1.0
    return eyc[numpy.searchsorted(ex, val)]

def less_equal_ecdf(x, val, cdf=()):
    """
    Given val return P(x ≤ val).

    Args:
        x (numpy.array): set of values
        val (float): value

    Returns:
        (float): probability that x ≤ val
    """
    x = numpy.asarray(x)
    if x.shape[0] == 0:
        return None
    if not cdf:
        ex, ey = ecdf(x)
    else:
        ex, ey = cdf
    # some short-circuit cases for discrete distributions
    if val > ex[-1]:
        return 1.0
    if val < ex[0]:
        return 0.0
    # uses numpy implementation of binary search
    return ey[numpy.searchsorted(ex, val, side='right') - 1]

def min_or_none(x):
    """
    Given an array x, returns the min value. If x = [], returns None.
    """
    if len(x) == 0:
        return None
    else:
        return numpy.min(x)

def max_or_none(x):
    """
    Given an array x, returns the max value. If x = [], returns None.
    """
    if len(x) == 0:
        return None
    else:
        return numpy.max(x)

def get_quantiles(sim_counts, obs_count):
    """ Computes delta1 and delta2 quantile scores from empirical distribution and observation """
    # delta 1 prob of observation at least n_obs events given the forecast
    delta_1 = greater_equal_ecdf(sim_counts, obs_count)
    # delta 2 prob of observing at most n_obs events given the catalog
    delta_2 = less_equal_ecdf(sim_counts, obs_count)
    return delta_1, delta_2

def poisson_log_likelihood(observation, forecast):
    """ Wrapper around scipy to compute the Poisson log-likelihood

    Args:
        observation: Observed (Grided) seismicity
        forecast: Forecast of a Model (Grided)

    Returns:
        Log-Liklihood values of between binned-observations and binned-forecasts
    """
    return numpy.log(scipy.stats.poisson.pmf(observation, forecast))

def poisson_joint_log_likelihood_ndarray(target_event_log_rates, target_observations, n_fore):
    """ Efficient calculation of joint log-likelihood of grid-based forecast.

    Note: log(w!) = 0

    Args:
        target_event_log_rates: natural log of bin rates where target events occurred
        target_observations: counts of target events
        n_fore: expected number from the forecasts

    Returns:
        joint_log_likelihood

    """
    sum_log_target_event_rates = numpy.sum(target_event_log_rates)
    # factorial(n) = loggamma(n+1)
    discrete_penalty_term = numpy.sum(scipy.special.loggamma(target_observations+1))
    return sum_log_target_event_rates - discrete_penalty_term - n_fore

def poisson_inverse_cdf(random_matrix, lam):
    """ Wrapper around scipy inverse poisson cdf function

    Args:
        random_matrix: Matrix of dimenions equal to forecast, containing random
                       numbers between 0 and 1.
        lam: vector of parameters for poisson distribution

    Returns:
        sample from the poisson distribution
    """
    return scipy.stats.poisson.ppf(random_matrix, lam)

def get_Kagan_I1_score(forecasts, catalog):
    """
    A program for scoring (I_1) earthquake-forecast grids by the methods of:
    Kagan, Yan Y. [2009] Testing long-term earthquake forecasts: likelihood methods
                         and error diagrams, Geophys. J. Int., v. 177, pages 532-542.

    Some advantages of these methods are that they:
        - are insensitive to the grid used to cover the Earth;
        - are insensitive to changes in the overall seismicity rate;
        - do not modify the locations or magnitudes of test earthquakes;
        - do not require simulation of virtual catalogs;
        - return relative quality measures, not just "pass" or "fail;" and
        - indicate relative specificity of forecasts as well as relative success.
    
    Written by Han Bao, UCLA, March 2021. Modified June 2021
    
    Note that: 
        (1) The testing catalog and forecast should have exactly the same time-window (duration)
        (2) Forecasts and catalogs have identical regions

    Args:
        forecasts:  csep.forecast or a list of csep.forecast (one catalog to test against different forecasts)
        catalog:    csep.catalog 
        
    Returns:
       I_1 (numpy.array): containing I1 for each forecast in inputs

    """
    ### Determine if input 'forecasts' is a list of csep.forecasts or a single csep.forecasts
    try:
        n_forecast = len(forecasts) # the input forecasts is a list of csep.forecast
    except:  
        n_forecast = 1             # the input forecasts is a single csep.forecast
        forecasts = [forecasts]

    # Sanity checks can go here
    for forecast in forecasts:
        if forecast.region != catalog.region:
            raise RuntimeError("Catalog and forecasts must have identical regions.")

    # Initialize array
    I_1 = numpy.zeros(n_forecast, dtype=numpy.float64)

    # Compute cell areas
    area_km2 = catalog.region.get_cell_area()
    total_area = numpy.sum(area_km2)
    
    for j, forecast in enumerate(forecasts):
        # Eq per cell per duration in forecast; note, if called on a CatalogForecast this could require computed expeted rates
        rate = forecast.spatial_counts()
        # Get Rate Density and uniform_forecast of the Forecast
        rate_den = rate / area_km2
        uniform_forecast = numpy.sum(rate) / total_area
        # Compute I_1 score
        n_event = catalog.event_count
        counts = catalog.spatial_counts()
        non_zero_idx = numpy.argwhere(rate_den > 0)
        non_zero_idx = non_zero_idx[:,0]
        I_1[j] = numpy.dot(counts[non_zero_idx], numpy.log2(rate_den[non_zero_idx] / uniform_forecast)) / n_event
    
    return I_1


def log_d_multinomial(x: numpy.ndarray, size: int, prob: numpy.ndarray):
    """

    Args:
        x:
        size:
        prob:

    Returns:

    """
    return scipy.special.loggamma(size + 1) + numpy.sum(
        x * numpy.log(prob) - scipy.special.loggamma(x + 1))


def MLL_score(union_catalog_counts: numpy.ndarray, catalog_counts: numpy.ndarray):
    """
    Calculates the modified Multinomial log-likelihood (MLL) score, defined by Serafini et al.,
    (2024). It is built from a collection catalogs Λ_u and a single catalog Ω

        MLL_score = 2 * log( L(Λ_u + N_u / N_o + Ω + 1) /
                           [L(Λ_u + N_u / N_o) * L(Ω + 1)]
                           )
    where N_u and N_j are the total number of events in Λ_u and Ω, respectively.

    Args:
        union_catalog_counts (numpy.ndarray):
        catalog_counts (numpy.ndarray):

    Returns:
        The MLL score for the collection of catalogs and
    """

    N_u = numpy.sum(union_catalog_counts)
    N_j = numpy.sum(catalog_counts)
    events_ratio = N_u / N_j

    union_catalog_counts_mod = union_catalog_counts + events_ratio
    catalog_counts_mod = catalog_counts + 1
    merged_catalog_j = union_catalog_counts_mod + catalog_counts_mod

    pr_merged_cat = merged_catalog_j / numpy.sum(merged_catalog_j)
    pr_union_cat = union_catalog_counts_mod / numpy.sum(union_catalog_counts_mod)
    pr_cat_j = catalog_counts_mod / numpy.sum(catalog_counts_mod)

    log_lik_merged = log_d_multinomial(x=merged_catalog_j,
                                       size=numpy.sum(merged_catalog_j),
                                       prob=pr_merged_cat)
    log_lik_union = log_d_multinomial(x=union_catalog_counts_mod,
                                      size=numpy.sum(union_catalog_counts_mod),
                                      prob=pr_union_cat)
    log_like_cat_j = log_d_multinomial(x=catalog_counts_mod,
                                       size=numpy.sum(catalog_counts_mod),
                                       prob=pr_cat_j)

    return 2 * (log_lik_merged - log_lik_union - log_like_cat_j)
