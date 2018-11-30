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
    ys = numpy.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def greater_equal_ecdf(x, val):
    """
    Given val return P(x ≥ val).

    Args:
        x (numpy.array): set of values
        val (float): value

    Returns:
        (float): probability that x ≤ val
    """
    x = numpy.asarray(x)
    ex, ey = ecdf(x)
    eyc = ey[::-1]
    # some short-circuit cases for discrete distributions
    if val > numpy.max(x):
        return 0.0
    if val < numpy.min(x):
        return 1.0
    return eyc[numpy.searchsorted(ex, val)]

def less_equal_ecdf(x, val):
    """
    Given val return P(x ≤ val).

    Args:
        x (numpy.array): set of values
        val (float): value

    Returns:
        (float): probability that x ≤ val
    """
    x = numpy.asarray(x)
    ex, ey = ecdf(x)

    # some short-circuit cases for discrete distributions
    if val > numpy.max(x):
        return 1.0
    if val < numpy.min(x):
        return 0.0
    return ey[numpy.searchsorted(ex,val,side='right')-1]
