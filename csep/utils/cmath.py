import numpy


def nearest_index(array, value):
    """
    Returns the index from array that is less than the value specified.
    """
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    """
    Returns the value from array that is less than the value specified.
    """
    array = numpy.asarray(array)
    idx = nearest_index(array, value)
    return array[idx]


def func_inverse(x, y, val, kind='nearest', **kwargs):
    """
    Returns the value of a function based on interpolation.
    """
    f = scipy.interpolate.interp1d(x, y, kind=kind, **kwargs)
    return f(val)

