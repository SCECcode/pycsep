# Third-party imports
import numpy
import scipy.interpolate

# PyCSEP imports
from csep.core.exceptions import CSEPException
from csep.utils.stats import binned_ecdf, sup_dist, get_quantiles
from csep.utils import flat_map_to_ndarray


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

def _get_tolerance(v):
    """Determine numerical tolerance due to limited precision of floating-point values.

    ... to account for roundoff error.

    In other words, returns a maximum possible difference that can be considered negligible.
    Only relevant for floating-point values.
    """
    if issubclass(v.dtype.type, numpy.floating):
        return numpy.abs(v) * numpy.finfo(v.dtype).eps
    return 0  # assuming it's an int

def bin1d_vec(p, bins, tol=None, right_continuous=False):
    """Efficient implementation of binning routine on 1D Cartesian Grid.

    Bins are inclusive on the lower bound
    and exclusive on the upper bound. In the case where a point does not fall within the bins a -1
    will be returned. The last bin extends to infinity when right_continuous is set as true.

    To correctly bin points that are practically on a bin edge, this function accounts for the
    limited precision of floating-point numbers (the roundoff error) with a numerical tolerance.
    If the provided points were subject to some floating-point operations after loading or
    generating them, the roundoff error increases (which is not accounted for) and requires
    overwriting the `tol` argument.

    Args:
        p (array-like): Point(s) to be placed into bins.
        bins (array-like): bin edges; must be sorted (monotonically increasing)
        tol (float): overwrite numerical tolerance, by default determined automatically from the
                     points' dtype to account for the roundoff error.
        right_continuous (bool): if true, consider last bin extending to infinity.

    Returns:
        numpy.ndarray: indices for the points corresponding to the bins.

    Raises:
        ValueError:
    """
    bins = numpy.asarray(bins)
    p = numpy.asarray(p)

    # if not np.all(bins[:-1] <= bins[1:]):  # check for sorted bins, which is a requirement
    #     raise ValueError("Bin edges are not sorted.")  # (pyCSEP always passes sorted bins)
    a0 = bins[0]
    if bins.size == 1:
        # for a single bin, set `right_continuous` to true; h is now arbitrary
        right_continuous = True
        h = 1.
    else:
        h = bins[1] - bins[0]

    if h < 0:
        raise ValueError("grid spacing must be positive and monotonically increasing.")

    # account for floating point precision
    a0_tol = _get_tolerance(a0)
    h_tol = a0_tol  # must be based on *involved* numbers
    p_tol = tol or _get_tolerance(p)

    idx = numpy.floor((p - a0 + p_tol + a0_tol) / (h - h_tol))
    idx = numpy.asarray(idx)  # assure idx is an array

    if right_continuous:
        # set upper bin index to last
        idx[idx < 0] = -1
        idx[idx >= len(bins) - 1] = len(bins) - 1
    else:
        idx[(idx < 0) | (idx >= len(bins))] = -1
    idx = idx.astype(numpy.int64)
    return idx

def _compute_likelihood(gridded_data, apprx_rate_density, expected_cond_count, n_obs):
    # compute pseudo likelihood
    idx = gridded_data != 0

    # this value is: -inf obs at idx and no apprx_rate_density
    #                -expected_cond_count if no target earthquakes
    n_events = numpy.sum(gridded_data)
    if n_events == 0:
        return(-expected_cond_count, numpy.nan)
    else:
        with numpy.errstate(divide='ignore'):
            likelihood = numpy.sum(gridded_data[idx] * numpy.log(apprx_rate_density[idx])) - expected_cond_count


    # cannot compute the spatial statistic score if there are no target events or forecast is computed undersampled
    if n_obs == 0 or expected_cond_count == 0:
        return (likelihood, numpy.nan)

    # normalize the rate density to sum to unity
    norm_apprx_rate_density = apprx_rate_density / numpy.sum(apprx_rate_density)

    # value could be: -inf if no value in apprx_rate_dens
    #                  nan if n_cat is 0
    with numpy.errstate(divide='ignore'):
        likelihood_norm = numpy.sum(gridded_data[idx] * numpy.log(norm_apprx_rate_density[idx])) / n_events

    return (likelihood, likelihood_norm)

def _compute_approximate_likelihood(gridded_data, apprx_forecasted_rate):
    """ Computes the approximate likelihood from Rhoades et al., 2011; Equation 4

    Args:
        gridded_data (ndarray): observed counts on spatial grid
        mean_rate_density (ndarray): mean rates from forecast

    Notes:
        Mean rates from the forecast are assumed to not have any zeros.
    """
    n_obs = numpy.sum(gridded_data)
    return numpy.sum(gridded_data*numpy.log10(apprx_forecasted_rate)) - n_obs

def _compute_spatial_statistic(gridded_data, log10_probability_map):
    """
    aggregates the log1
    Args:
        gridded_data:
        log10_probability_map:
    """
    # returns a unique set of indexes corresponding to cells where earthquakes occurred
    # this should implement similar logic to the spatial tests wrt undersampling.
    # techincally, if there are are target eqs you can't compute this statistic.
    if numpy.sum(gridded_data) == 0:
        return numpy.nan
    idx = numpy.unique(numpy.argwhere(gridded_data))
    return numpy.sum(log10_probability_map[idx])

def _distribution_test(stochastic_event_set_data, observation_data):

    # for cached files want to write this with memmap
    union_catalog = flat_map_to_ndarray(stochastic_event_set_data)
    min_time = 0.0
    max_time = numpy.max([numpy.max(numpy.ceil(union_catalog)), numpy.max(numpy.ceil(observation_data))])

    # build test_distribution with 100 data points
    num_points = 100
    tms = numpy.linspace(min_time, max_time, num_points, endpoint=True)

    # get combined ecdf and obs ecdf
    combined_ecdf = binned_ecdf(union_catalog, tms)
    obs_ecdf = binned_ecdf(observation_data, tms)

    # build test distribution
    n_cat = len(stochastic_event_set_data)
    test_distribution = []
    for i in range(n_cat):
        test_ecdf = binned_ecdf(stochastic_event_set_data[i], tms)
        # indicates there were zero events in catalog
        if test_ecdf is not None:
            d = sup_dist(test_ecdf[1], combined_ecdf[1])
            test_distribution.append(d)
    d_obs = sup_dist(obs_ecdf[1], combined_ecdf[1])

    # score evaluation
    _, quantile = get_quantiles(test_distribution, d_obs)

    return test_distribution, d_obs, quantile

def cleaner_range(start, end, h):
    """ Returns array holding bin edges that doesn't contain floating point wander.

    Floating point wander can occur when repeatedly adding floating point numbers together. The errors propogate and become worse over the sum. This function generates the
    values on an integer grid and converts back to floating point numbers through multiplication.

     Args:
        start (float)
        end (float)
        h (float): magnitude spacing

    Returns:
        bin_edges (numpy.ndarray)
    """
    # determine required scaling to account for decimal places of bin edges and stepping
    num_decimals_bins = len(str(float(start)).split('.')[1])
    scale = max(10**num_decimals_bins, 1 / h)
    start = numpy.round(scale * start)
    end = numpy.round(scale * end)
    d = scale * h
    return numpy.arange(start, end + d / 2, d) / scale

def first_nonnan(arr, axis=0, invalid_val=-1):
    mask = arr==arr
    return numpy.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonnan(arr, axis=0, invalid_val=-1):
    mask = arr==arr
    val = arr.shape[axis] - numpy.flip(mask, axis=axis).argmax(axis=axis) - 1
    return numpy.where(mask.any(axis=axis), val, invalid_val)