import numpy
import numpy as np
import scipy.interpolate
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

def discretize(data, bin_edges, right_continuous=False):
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
    idx = bin1d_vec(data, bin_edges, right_continuous=right_continuous)
    if numpy.any(idx == -1):
        raise CSEPException("Discretized values should all be within bin_edges")
    x_new = bin_edges[idx]
    return x_new

def bin1d_vec(p, bins, right_continuous=False):
    """Efficient implementation of binning routine on 1D Cartesian Grid.

    Returns the indices of the points into bins. Bins are inclusive on the lower bound
    and exclusive on the upper bound. In the case where a point does not fall within the bins a -1
    will be returned. The last bin extends to infinity.

    Args:
        p (array-like): Point(s) to be placed into b
        bins (array-like): bins to considering for binning, must be monotonically increasing
        right_continuous (bool): if true, consider last bin extending to infinity

    Returns:
        idx (array-like): indexes hashed into grid

    Raises:
        ValueError:
    """
    a0 = numpy.min(bins)
    h = bins[1] - bins[0]
    eps = numpy.finfo(np.float).eps
    if h < 0:
        raise ValueError("grid spacing must be positive.")
    idx = numpy.floor((p + eps - a0) / h)
    if right_continuous:
        # set upper bin index to last
        idx[(idx >= len(bins) - 1)] = len(bins) - 1
        idx[(idx < 0)] = -1
    else:
        # if outside set to nan
        idx[((idx < 0) | (idx >= len(bins) - 1))] = -1
    idx = idx.astype(numpy.int)

    return idx
