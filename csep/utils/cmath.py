import numpy
import scipy


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

def discretize(data, bin_edges):
    """
    returns array with len(bin_edges) consisting of the discretized values from each bin.
    instead of returning the counts of each bin, this will return an array with values
    modified such that any value within bin_edges[0] <= x_new < bin_edges[1] ==> bin_edges[0].
    """
    bin_edges = numpy.array(bin_edges)
    if bin_edges[1] < bin_edges[0]:
        raise ValueError("bin_edges must be increasing.")
    data = numpy.array(data)
    idx = numpy.digitize(data, bins=bin_edges)
    x_new = bin_edges[idx]
    return x_new