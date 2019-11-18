import numpy
import numpy as np
import scipy
from csep.core.exceptions import CSEPException


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

    This implementation forces you to define a bin edge that contains the data.
    """
    bin_edges = numpy.array(bin_edges)
    if bin_edges.size == 0:
        raise ValueError("bin_edges must not be empty")
    if bin_edges[1] < bin_edges[0]:
        raise ValueError("bin_edges must be increasing")
    data = numpy.array(data)
    idx = bin1d_vec(data, bin_edges)
    if numpy.any(idx == -1):
        raise CSEPException("Discretized values should all be within bin_edges")
    x_new = bin_edges[idx]
    return x_new

def bin1d_vec(p, bins):
    """
    same as bin1d but optimized for vectorized calls.
    """
    a0 = np.min(bins)
    h = bins[1] - bins[0]
    eps = np.finfo(np.float).eps
    idx = np.floor((p+eps-a0)/h)
    idx[((idx < 0) | (idx >= len(bins)-1))] = -1
    return idx.astype(np.int)
