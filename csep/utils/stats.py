import numpy


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
    if not cdf:
        ex, ey = ecdf(x)
    else:
        ex, ey = cdf

    # some short-circuit cases for discrete distributions
    if val > ex[-1]:
        return 1.0
    if val < ex[0]:
        return 0.0
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
