import numpy


def sup_dist(cdf1, cdf2):
    """
    given two cumulative distribution functions, compute the supremum of the set of absolute distances.

    note:
        this function does not check that the ecdfs are ordered or balanced. beware!
    """
    return numpy.max(numpy.absolute(cdf2 - cdf1))

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
    """
    Direct call using ndarray. Useful for optimizing calls to multiprocessing pools.
    """
    # delta 1 prob of observation at least n_obs events given the forecast
    delta_1 = greater_equal_ecdf(sim_counts, obs_count)
    # delta 2 prob of observing at most n_obs events given the catalog
    delta_2 = less_equal_ecdf(sim_counts, obs_count)
    return delta_1, delta_2