import copy
from collections import namedtuple, defaultdict
import os
from tempfile import mkstemp

import numpy
import tqdm

import time
from csep.utils.stats import cumulative_square_dist, binned_ecdf, sup_dist
from csep.utils.constants import CSEP_MW_BINS, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_WEEK
from csep.utils import flat_map_to_ndarray
from csep.utils.plotting import plot_magnitude_test, plot_number_test, plot_spatial_test, \
    plot_cumulative_events_versus_time_dev, plot_magnitude_histogram_dev, plot_spatial_dataset, plot_distribution_test, \
    plot_likelihood_test

# implementing plotting routines as functions
from csep.utils.stats import get_quantiles
from csep.utils.spatial import bin1d_vec

EvaluationResult = namedtuple('EvaluationResult', ['test_distribution',
                                                   'name',
                                                   'observed_statistic',
                                                   'quantile',
                                                   'status',
                                                   'obs_catalog_repr',
                                                   'sim_name',
                                                   'obs_name'])

def number_test(stochastic_event_sets, observation, event_counts=None):
    # get number of events for observations and simulations
    if not event_counts:
        sim_counts = []
        for catalog in stochastic_event_sets:
            sim_counts.append(catalog.event_count)
    else:
        sim_counts = event_counts
    observation_count = observation.event_count
    # get delta_1 and delta_2 values
    delta_1, delta_2 = get_quantiles(sim_counts, observation_count)
    # prepare result
    result = EvaluationResult(test_distribution=sim_counts,
                              name='N-Test',
                              observed_statistic=observation_count,
                              quantile=(delta_1, delta_2),
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result

def pseudo_likelihood_test(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    """
    This test makes a discrete approximation of the continuous conditional rate-density function
    that governs the forecast. The intensity is magnitude independent, so only varies spatially and not
    with magnitude. This provides a scalar conditional intensity value in each cell.

    We build the distribution of the test statistic under the null hypothesis by computing the psuedo-likelihood score for
    each catalog given the approximate conditional intensity function, by summing likelihoods over each event in the set.
    We normalize this by the expected number of events from the conditional intensity function.

    Note: Some of these statements are vectorized for performance.

    If writing a custom reduce_func, this needs to reduce of axis=1 of a numpy.ndarray. For example, c = numpy.mean(a, axis=1).

    Args:
        stochastic_event_sets: list of catalogs
        observation: catalog
        apprx_log_rate_density: 1d array corresponding to region. density should be counts / dh / dh / time_interval (yr)
        normed: this normalizes by the cell event-counts to isolate the spatial distribution, effectively the s-test

    Returns:
        p-value: float
    """

    # integrating, assuming that all cats in ses have same region
    region = stochastic_event_sets[0].region
    expected_cond_count = numpy.sum(apprx_rate_density) * region.dh * region.dh * time_interval
    gridded_obs = observation.gridded_event_counts()
    name = 'L-Test'

    # build likelihood distribution from ses
    test_distribution = []
    t0 = time.time()
    for i, catalog in enumerate(stochastic_event_sets):
        gridded_cat = catalog.gridded_event_counts()
         # compute likelihood for each event, ignoring areas with 0 expectation
        gridded_cat_ma = numpy.ma.masked_where(gridded_cat == 0, gridded_cat)
        apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_cat_ma.mask)
        likelihood = numpy.ma.sum(gridded_cat_ma * numpy.ma.log10(apprx_rate_density_ma)) - expected_cond_count
        test_distribution.append(likelihood)
        # report real-time stats
        if (i+1) % 5000 == 0:
            t1 = time.time()
            print(f"Processed {i+1} catalogs in {t1-t0} seconds.")

    # compute psuedo-likelihood for comcat
    gridded_obs_ma = numpy.ma.masked_where(gridded_obs == 0, gridded_obs)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_obs_ma.mask)
    comcat_likelihood = numpy.ma.sum(gridded_obs_ma * numpy.ma.log10(apprx_rate_density_ma)) - expected_cond_count

    # determine outcome of evaluation, check for infinity
    _, quantile = get_quantiles(test_distribution, comcat_likelihood)

    # Signals outcome of test
    message = "Normal"

    # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
    # either normal and wrong or udetermined (undersampled)
    if numpy.isclose(quantile, 0.0) or numpy.isclose(quantile, 1.0):

        # undetermined failure of the test
        if numpy.isinf(comcat_likelihood):

            # Build message
            message = f"undetermined. Infinite likelihood scores found."

    # build evaluation result
    result = EvaluationResult(test_distribution=test_distribution,
                              name=name,
                              observed_statistic=comcat_likelihood,
                              quantile=quantile,
                              status=message,
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def spatial_test(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    """
    Computes an evaluation of a forecast spatially by isolating the spatial distribution of the approximate conditional
    rate density. It does this by normalizing the expected conditional rate density by the ratio between observed and
    simulated event counts. This effectively removes the rate component from the forecast.

    Args:
        stochastic_event_sets: list of catalogs
        observation: catalog
        apprx_rate_density: 1d array corresponding to region,

    Returns:
        quantile: p(X ≤ x) from the test distribution and observation

    """
    # integrating, assuming that all cats in ses have same region
    name = 'S-Test'

    # build likelihood distribution from ses
    region = stochastic_event_sets[0].region
    test_distribution = []
    # this could be io based if iterator is passed
    for catalog in tqdm.tqdm(stochastic_event_sets, total=len(stochastic_event_sets)):
        gridded_rate_cat = catalog.gridded_event_counts() / time_interval
        # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio
        normalizing_factor = observation.event_count / catalog.event_count
        gridded_rate_cat_norm = normalizing_factor * gridded_rate_cat
        # compute likelihood for each event, ignoring areas with 0 expectation
        gridded_rate_cat_norm_ma = numpy.ma.masked_where(gridded_rate_cat_norm == 0, gridded_rate_cat_norm)
        apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_rate_cat_norm_ma.mask)
        likelihood = numpy.ma.sum(gridded_rate_cat_norm_ma * numpy.ma.log10(apprx_rate_density_ma))
        test_distribution.append(likelihood)

    # compute psuedo-likelihood for comcat
    gridded_obs_rate = observation.gridded_event_counts() / time_interval
    gridded_obs_rate_ma = numpy.ma.masked_where(gridded_obs_rate == 0, gridded_obs_rate)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_obs_rate_ma.mask)
    comcat_likelihood = numpy.ma.sum(gridded_obs_rate_ma * numpy.ma.log10(apprx_rate_density_ma))

    # determine outcome of evaluation, check for infinity
    _, quantile = get_quantiles(test_distribution, comcat_likelihood)

    # Signals outcome of test
    message = "Normal"
    # Deal with case with cond. rate. density func has zeros
    if numpy.isclose(quantile, 0.0) or numpy.isclose(quantile, 1.0):
        # undetermined failure of the test
        if numpy.isinf(comcat_likelihood):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."
    # build evaluation result
    result = EvaluationResult(test_distribution=test_distribution,
                              name=name,
                              observed_statistic=comcat_likelihood,
                              quantile=quantile,
                              status=message,
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result

def magnitude_test(stochastic_event_sets, observation, mag_bins=CSEP_MW_BINS):
    """
    Compares the observed magnitude distribution with the forecasted magnitude distribution. It does this by generating a
    union distribution (distribution of magnitudes for all events in all stochastic event sets). Then we can compare the
    magnitude distribution in each individual stochastic event set to build the test distribution under the null hypothesis.
    Like the other tests, we report the p-value of the observation given the distribution of the test statistics under the
    null hypothesis.

    Right now this evaluations scales by the ratio of the event counts in the stochastic event sets and the observed catalog.
    We randomly sample N_Obs_Events to ensure the variance of the test distribution is consistent with that of the observations.


    Args:
        stochastic_event_set: list of catalogs
        observation: observation catalog

    Returns:
        quantile: quantile score P(X ≤ x)
    """
    # get data that we need
    stochastic_event_sets_magnitudes = [cat.get_magnitudes() for cat in stochastic_event_sets]
    union_catalog = flat_map_to_ndarray(stochastic_event_sets_magnitudes)
    n_union_events = len(union_catalog)

    # build bins to compute histograms from experiment details

    obs_magnitudes = observation.get_magnitudes()
    n_obs_events = len(obs_magnitudes)

    # normalize by event counts
    scale = n_obs_events / n_union_events
    union_histogram = numpy.histogram(union_catalog, bins=mag_bins)[0] * scale

    # compute histograms and convert to rates
    obs_histogram, bin_edges = numpy.histogram(obs_magnitudes, bins=mag_bins)

    # build test-distribution
    test_distribution = []
    for ses_mags in stochastic_event_sets_magnitudes:
        n_ses_mags = len(ses_mags)
        if n_ses_mags == 0:
            scale = 0
        else:
            scale = n_obs_events / n_ses_mags
        ses_histogram = numpy.histogram(ses_mags, bins=mag_bins)[0] * scale
        # this distribution might not have the expected variance given n_obs_events.
        d_statistic = cumulative_square_dist(ses_histogram, union_histogram)
        test_distribution.append(d_statistic)

    # compute statistic from the observation
    obs_d_statistic = cumulative_square_dist(obs_histogram, union_histogram)

    # score evaluation
    _, quantile = get_quantiles(test_distribution, obs_d_statistic)

    result = EvaluationResult(test_distribution=test_distribution,
                              name='M-Test',
                              observed_statistic=obs_d_statistic,
                              quantile=quantile,
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def _compute_likelihood(gridded_data, apprx_rate_density, expected_cond_count, n_obs):
    """
    not sure if this should actually be used masked arrays, bc we are losing information about undetermined l-test results.
    apply spatial smoothing here?
    """
    gridded_cat_ma = numpy.ma.masked_where(gridded_data == 0, gridded_data)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_cat_ma.mask)
    likelihood = numpy.sum(gridded_cat_ma * numpy.log10(apprx_rate_density_ma)) - expected_cond_count
    # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio, this should never be 0, else forecast is expecting 0 earthquakes in region.
    normalizing_factor = n_obs / expected_cond_count
    normed_rate_density_ma = normalizing_factor * apprx_rate_density_ma
    # compute likelihood for each event, ignoring cells with 0 events in the catalog.
    likelihood_norm = numpy.sum(gridded_cat_ma * numpy.ma.log10(normed_rate_density_ma)) / numpy.sum(gridded_cat_ma)
    return(likelihood, likelihood_norm)

def combined_likelihood_and_spatial(stochastic_event_sets, observation, apprx_rate_density, time_interval=1.0):
    # integrating, assuming that all cats in ses have same region
    region = stochastic_event_sets[0].region
    n_cat = len(stochastic_event_sets)
    expected_cond_count = numpy.sum(apprx_rate_density) * region.dh * region.dh * time_interval
    gridded_obs = observation.gridded_event_counts()
    n_obs = observation.get_number_of_events()

    # build likelihood distribution from ses
    test_distribution_likelihood = numpy.empty(n_cat)
    test_distribution_spatial = numpy.empty(n_cat)
    t0 = time.time()
    for i, catalog in enumerate(stochastic_event_sets):
        gridded_cat = catalog.gridded_event_counts()
        # compute likelihood for each event, ignoring areas with 0 expectation,
        lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density, expected_cond_count, n_obs)
        # store results
        test_distribution_likelihood[i] = lh
        test_distribution_spatial[i] = lh_norm
        if (i+1) % 500 == 0:
            t1 = time.time()
            print(f"Processed {i+1} catalogs in {t1-t0} seconds.")

    obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density, expected_cond_count, n_obs)

    # determine outcome of evaluation, check for infinity
    _, quantile_likelihood = get_quantiles(test_distribution_likelihood, obs_lh)
    _, quantile_spatial = get_quantiles(test_distribution_spatial, obs_lh_norm)

    # Signals outcome of test
    message = "Normal"
    # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
    # either normal and wrong or udetermined (undersampled)
    if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
        # undetermined failure of the test
        if numpy.isinf(obs_lh):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."
    # build evaluation result
    result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood,
                                         name='L-Test',
                                         observed_statistic=obs_lh,
                                         quantile=quantile_likelihood,
                                         status=message,
                                         obs_catalog_repr=str(observation),
                                         sim_name=stochastic_event_sets[0].name,
                                         obs_name=observation.name)
    # find out if there are issues with the test
    if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
        # undetermined failure of the test
        if numpy.isinf(obs_lh_norm):
            # Build message
            message = f"undetermined. Infinite likelihood scores found."

    # build evaluation result
    result_spatial = EvaluationResult(test_distribution=test_distribution_spatial,
                                      name='S-Test',
                                      observed_statistic=obs_lh_norm,
                                      quantile=quantile_spatial,
                                      status=message,
                                      obs_catalog_repr=str(observation),
                                      sim_name=stochastic_event_sets[0].name,
                                      obs_name=observation.name)
    return (result_likelihood, result_spatial)

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

def interevent_time_test(stochastic_event_sets, observation):
    """
    These compare the inter-event time distribution of the forecasts with the observation. It works similarly to
    the magnitude test. First, we build the union distribution

    Args:
        stochastic_event_sets:
        observation:

    Returns:

    """
    # get data that we need
    inter_event_times = [cat.get_inter_event_times() for cat in stochastic_event_sets]

    # get inter-event times from catalog
    obs_times = observation.get_inter_event_times()

    # compute distribution statistics
    test_distribution, d_obs, quantile = _distribution_test(inter_event_times, obs_times)

    result = EvaluationResult(test_distribution=test_distribution,
                              name='IETD-Test',
                              observed_statistic=d_obs,
                              quantile=quantile,
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def interevent_distance_test(stochastic_event_sets, observation):
    # get data that we need
    inter_event_distances = [cat.get_inter_event_distances() for cat in stochastic_event_sets]

    # get inter-event times from catalog
    obs_times = observation.get_inter_event_distances()

    # compute distribution statistics
    test_distribution, d_obs, quantile = _distribution_test(inter_event_distances, obs_times)

    result = EvaluationResult(test_distribution=test_distribution,
                              name='IESD-Test',
                              observed_statistic=d_obs,
                              quantile=quantile,
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def total_event_rate_distribution_test(stochastic_event_sets, observation):
    # get data that we need
    terd = [cat.gridded_event_counts() for cat in stochastic_event_sets]

    # get inter-event times from catalog
    obs_terd = observation.gridded_event_counts()

    # compute distribution statistics
    test_distribution, d_obs, quantile = _distribution_test(terd, obs_terd)

    result = EvaluationResult(test_distribution=test_distribution,
                              name='TERD-Test',
                              observed_statistic=d_obs,
                              quantile=quantile,
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)

    return result

def bvalue_test(stochastic_event_sets, observation):
    # get number of events for observations and simulations
    sim_counts = []
    for catalog in stochastic_event_sets:
        bv = catalog.get_bvalue()
        if bv is not None:
            sim_counts.append(bv)
    observation_count = observation.get_bvalue()
    # get delta_1 and delta_2 values
    _, quantile = get_quantiles(sim_counts, observation_count)
    # prepare result
    result = EvaluationResult(test_distribution=sim_counts,
                              name='BV-Test',
                              observed_statistic=observation_count,
                              quantile=quantile,
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result

class AbstractProcessingTask:
    def __init__(self, data=None, name=None):
        self.data = data or []
        self.mws = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.name = name
        self.ax = []
        self.fnames = []
        self.needs_two_passes = False
        self.cache = False
        self.buffer = []
        self.region = None
        self.buffer_fname = None
        self.fhandle = None

    @staticmethod
    def _build_figure_filename(dir, mw, plot_id):
        basename = f"{plot_id}_{mw}_test"
        return os.path.join(dir, basename)

    @staticmethod
    def _get_temporary_filename():
        # create temporary file and return filename
        _, tmp_file = mkstemp()
        return tmp_file

    def process_catalog(self, catalog):
        raise NotImplementedError('must implement process_catalog()!')

    def evaluate(self, obs, args=None):
        """
        Compute evaluation of data stored in self.data.

        Args:
            obs (csep.Catalog): used to evaluate the forecast
            args (tuple): args for this function

        Returns:
            result (csep.core.evaluations.EvaluationResult):

        """
        raise NotImplementedError('must implement evaluate()!')

    def plot(self, results, plot_dir, show=False):
        """
        plots function, typically just a wrapper to function in utils.plotting()

        Args:
            show (bool): show plot, if plotting multiple, just run on last.
            filename (str): where to save the file
            plot_args (dict): plotting args to pass to function

        Returns:
            axes (matplotlib.axes)

        """
        raise NotImplementedError('must implement plot()!')

    def process_again(self, catalog, args=()):
        """ This function defaults to pass unless the method needs to read through the data twice. """
        pass

    def cache_results(self, results, buf_len=1000):
        self.buf_len = buf_len
        self.buffer.append(results)
        if len(self.buffer) >= buf_len:
            out = numpy.array(self.buffer)
            out.tofile(self.fhandle)
            self.buffer = []

    def store_results(self, fname):
        """ archive results of calcuations to compute comparisons plots. """
        pass

class NumberTest(AbstractProcessingTask):

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            counts.append(cat_filt.event_count)
        self.data.append(counts)

    def evaluate(self, obs, args=None):
        # we dont need args for this function
        _ = args
        results = {}
        data = numpy.array(self.data)
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            observation_count = obs_filt.event_count
            # get delta_1 and delta_2 values
            delta_1, delta_2 = get_quantiles(data[:,i], observation_count)
            # prepare result
            result = EvaluationResult(test_distribution=data[:,i],
                          name='N-Test',
                          observed_statistic=observation_count,
                          quantile=(delta_1, delta_2),
                          status='Normal',
                          obs_catalog_repr=str(obs),
                          sim_name=self.name,
                          obs_name=obs.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):

        for mw, result in results.items():
            # compute bin counts, this one is special because of integer values
            td = result.test_distribution
            min_bin, max_bin = numpy.min(td), numpy.max(td)
            # hard-code some logic for bin size
            if mw < 4.0:
                bins = 'sqrt'
            else:
                bins = numpy.arange(min_bin, max_bin)
            n_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'n_test')
            ax = plot_number_test(result, show=show, plot_args={'percentile': 95,
                                      'title': f'Number-Test\nMw>{mw}',
                                      'bins': 'auto',
                                      'filename': n_test_fname})
            self.fnames.append(n_test_fname)

class MagnitudeTest(AbstractProcessingTask):

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        # always compute this for the lowest magnitude, above this is redundant
        mags = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            binned_mags = cat_filt.binned_magnitude_counts()
            mags.append(binned_mags)
        # data shape (n_cat, n_mw, n_mw_bins)
        self.data.append(mags)

    def evaluate(self, obs, args=None):
        # we dont need args
        _ = args
        results = {}
        for i, mw in enumerate(self.mws):
            test_distribution = []
            # get observed magnitude counts
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            # shape (n_mw_bins)
            obs_histogram = obs_filt.binned_magnitude_counts()
            n_obs_events = numpy.sum(obs_histogram)
            mag_counts_all = numpy.array(self.data)
            # get the union histogram, simply the sum over all catalogs, (n_cat, n_mw)
            n_union_events = numpy.sum(mag_counts_all[:,i,:])
            scale = n_obs_events / n_union_events
            union_histogram = numpy.sum(mag_counts_all[:,i,:], axis=0) * scale
            # could do this without a loop, c'est la.
            for j in range(mag_counts_all.shape[0]):
                n_events = numpy.sum(mag_counts_all[j,i,:])
                if n_events == 0:
                    scale = 0
                else:
                    scale = n_obs_events / n_events
                catalog_histogram = mag_counts_all[j,i,:] * scale
                test_distribution.append(cumulative_square_dist(catalog_histogram, union_histogram))
            # compute statistic from the observation
            obs_d_statistic = cumulative_square_dist(obs_histogram, union_histogram)
            # score evaluation
            _, quantile = get_quantiles(test_distribution, obs_d_statistic)
            # prepare result
            result = EvaluationResult(test_distribution=test_distribution,
                                      name='M-Test',
                                      observed_statistic=obs_d_statistic,
                                      quantile=quantile,
                                      status='Normal',
                                      obs_catalog_repr=str(obs),
                                      obs_name=obs.name,
                                      sim_name=self.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, show=False):
        # get the filename
        for mw in self.mws:
            m_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'm-test')
            plot_args = {'percentile': 95,
                 'title': f'Magnitude-Test\nMw>{mw}',
                  'bins': 'auto',
                  'filename': m_test_fname}
            ax = plot_magnitude_test(results[mw], show=False, plot_args=plot_args)
        # self.ax.append(ax)
            self.fnames.append(m_test_fname)

class NumberTestDelayed(AbstractProcessingTask):

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            counts.append(cat_filt.event_count)
        return numpy.array(counts)

    def evaluate_test(self, data, obs, args=None):
        # we dont need args for this function
        _ = args
        results = {}
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            observation_count = obs_filt.event_count
            # get delta_1 and delta_2 values
            delta_1, delta_2 = get_quantiles(data[:,i], observation_count)
            # prepare result
            result = EvaluationResult(test_distribution=data[:,i],
                          name='N-Test',
                          observed_statistic=observation_count,
                          quantile=(delta_1, delta_2),
                          status='Normal',
                          obs_catalog_repr=str(obs),
                          sim_name=self.name,
                          obs_name=obs.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, show=False):
        for mw, result in results.items():
            # compute bin counts, this one is special because of integer values
            td = result.test_distribution
            min_bin, max_bin = numpy.min(td), numpy.max(td)
            # hard-code some logic for bin size
            if mw < 4.0:
                bins = 'auto'
            else:
                bins = numpy.arange(min_bin, max_bin)
            n_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'n_test')
            ax = plot_number_test(result, show=show,
                           plot_args={'percentile': 95,
                                      'title': f'Number-Test\nMw>{mw}',
                                      'bins': bins,
                                      'filename': n_test_fname})
            # self.ax.append(ax)
            self.fnames.append(n_test_fname)

class LikelihoodAndSpatialTest(AbstractProcessingTask):
    def __init__(self, cache=True, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution_spatial = []
        self.test_distribution_likelihood = []
        self.cat_id = 0
        self.needs_two_passes = True
        self.cache = cache
        self.buffer = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []
        if self.cache:
            self.needs_two_passes = False

    def __del__(self):
        print('Removing temporary file.')
        if self.cache and self.buffer_fname:
            os.remove(self.buffer_fname)
        if self.fhandle:
            self.fhandle.close()

    def process_catalog(self, catalog):
        # grab stuff from catalog that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from catalog
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            gridded_counts = cat_filt.gridded_event_counts()
            counts.append(gridded_counts)
        if self.cache:
            if self.buffer_fname is None:
                self.buffer_fname = self._get_temporary_filename()
                self.fhandle = open(self.buffer_fname, 'wb+')
            self.cache_results(counts)
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the files
        if self.cache:
            return
        time_horizon, n_cat, end_epoch, obs = args
        apprx_rate_density = self.data / self.region.dh / self.region.dh / time_horizon / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1) * self.region.dh * self.region.dh * time_horizon
        # unfortunately, we need to iterate twice through the catalogs for this.
        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}')
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude > {mw}')
            gridded_cat = cat_filt.gridded_event_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm
        self.test_distribution_likelihood.append(lhs)
        self.test_distribution_spatial.append(lhs_norm)

    def evaluate(self, obs, args=None):
        """ this function is slow, it could use some love eventually. """
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}
        items_per_read = len(self.mws) * self.region.num_nodes
        apprx_rate_density = self.data / self.region.dh / self.region.dh / time_horizon / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1) * self.region.dh * self.region.dh * time_horizon
        size_of_float = numpy.dtype(numpy.float64).itemsize
        # build test distribution from file buffer
        if self.cache:
            # flush buffer if reading failed for some reason
            if len(self.buffer) != 0:
                print(f'Found {len(self.buffer)} items still in buffer. Flushing buffer before continuing.')
                out = numpy.array(self.buffer)
                out.tofile(self.fhandle)
                self.buffer = []
                self.fhandle.close()
            # make sure things are consistent
            nchunks = int(n_cat / self.buf_len)
            stragglers = n_cat % self.buf_len
            assert nchunks*self.buf_len + stragglers == n_cat
            t0 = time.time()
            print(nchunks, stragglers, n_cat, self.buf_len)
            for j in range(nchunks+1):
                if j < nchunks:
                    read_buf_len = self.buf_len
                    gridded_cat_mws = numpy.fromfile(self.buffer_fname,
                                                 count=items_per_read*self.buf_len,
                                                 offset=j*size_of_float*items_per_read*self.buf_len) \
                    .reshape(self.buf_len, len(self.mws), self.region.num_nodes)
                # this happens only once
                elif stragglers !=0:
                    read_buf_len = stragglers
                    gridded_cat_mws = numpy.fromfile(self.buffer_fname,
                                                 count=items_per_read*read_buf_len,
                                                 offset=j*size_of_float*items_per_read*self.buf_len) \
                    .reshape(read_buf_len, len(self.mws), self.region.num_nodes)
                # loop over catalog data
                for k in range(read_buf_len):
                    lhs = numpy.zeros(len(self.mws))
                    lhs_norm = numpy.zeros(len(self.mws))
                    for i, mw in enumerate(self.mws):
                        obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
                        n_obs = obs_filt.event_count
                        gridded_cat = gridded_cat_mws[k, i, :]
                        lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i, :],
                                                          expected_cond_count[i], n_obs)
                        lhs[i] = lh
                        lhs_norm[i] = lh_norm
                    self.test_distribution_likelihood.append(lhs)
                    self.test_distribution_spatial.append(lhs_norm)

                    if (j*self.buf_len+(k+1)) % 2500 == 0:
                        t1 = time.time()
                        print(f'Processed {j*self.buf_len+k+1} in {t1 - t0} seconds')

                if (j+1)*self.buf_len % n_cat == 0:
                    break

        test_distribution_likelihood = numpy.array(self.test_distribution_likelihood)
        test_distribution_spatial = numpy.array(self.test_distribution_spatial)
        # prepare results for each mw
        for i, mw in enumerate(self.mws):
            # get observed likelihood
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.gridded_event_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            # determine outcome of evaluation, check for infinity
            _, quantile_likelihood = get_quantiles(test_distribution_likelihood[:,i], obs_lh)
            _, quantile_spatial = get_quantiles(test_distribution_spatial[:,i], obs_lh_norm)
            # Signals outcome of test
            message = "Normal"
            # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
            # either normal and wrong or udetermined (undersampled)
            if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh):
                    # Build message
                    message = f"undetermined. Infinite likelihood scores found."
            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:,i],
                                                 name='L-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 obs_catalog_repr=str(obs),
                                                 sim_name=self.name,
                                                 obs_name=obs.name)
            # find out if there are issues with the test
            if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh_norm):
                    # Build message
                    message = f"undetermined. Infinite likelihood scores found."

            # build evaluation result
            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial[:,i],
                                              name='S-Test',
                                              observed_statistic=obs_lh_norm,
                                              quantile=quantile_spatial,
                                              status=message,
                                              obs_catalog_repr=str(obs),
                                              sim_name=self.name,
                                              obs_name=obs.name)

            results[mw] = (result_likelihood, result_spatial)

        return results

    def plot(self, results, plot_dir, show=False):
        for mw, result_tuple in results.items():
            # plot spatial test
            s_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 's-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test\nMw>{mw}',
                         'bins': 'auto',
                         'filename': s_test_fname}
            spatial_ax = plot_spatial_test(result_tuple[1], axes=None, plot_args=plot_args, show=False)
            # plot likelihood test
            l_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'l-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test\nMw>{mw}',
                         'bins': 'auto',
                         'filename': l_test_fname}
            ax = plot_likelihood_test(result_tuple[0], axes=None, plot_args=plot_args, show=show)
            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['l-test'].append(l_test_fname)
            self.fnames['s-test'].append(s_test_fname)

class CumulativeEventPlot(AbstractProcessingTask):

    def __init__(self, origin_epoch, end_epoch, **kwargs):
        super().__init__(**kwargs)
        self.origin_epoch = origin_epoch
        self.end_epoch = end_epoch
        self.time_bins, self.dt = self._get_time_bins()
        self.n_bins = self.time_bins.shape[0]

    def _get_time_bins(self):
        diff = (self.end_epoch - self.origin_epoch) / SECONDS_PER_DAY / 1000
        # if less than 7 day use hours
        if diff <= 7.0:
            dt = SECONDS_PER_HOUR * 1000
        # if less than 180 day use days
        elif diff <= 180:
            dt = SECONDS_PER_DAY * 1000
        # if less than 3 years (1,095.75 days) use weeks
        elif diff <= 1095.75:
            dt = SECONDS_PER_WEEK * 1000
        # use 30 day
        else:
            dt = SECONDS_PER_DAY * 1000 * 30
        # always make bins from start to end of catalog
        return numpy.arange(self.origin_epoch, self.end_epoch+dt/2, dt), dt

    def process_catalog(self, catalog):
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            n_events = cat_filt.catalog.shape[0]
            ses_origin_time = cat_filt.get_epoch_times()
            inds = bin1d_vec(ses_origin_time, self.time_bins)
            binned_counts = numpy.zeros(self.n_bins)
            for j in range(n_events):
                binned_counts[inds[j]] += 1
            counts.append(binned_counts)
        self.data.append(counts)

    def evaluate(self, obs, args=None):
        # data are stored as (n_cat, n_mw_bins, n_time_bins)
        summed_counts = numpy.cumsum(self.data, axis=2)
        # compute summary statistics for plotting
        fifth_per = numpy.percentile(summed_counts, 5, axis=0)
        first_quar = numpy.percentile(summed_counts, 25, axis=0)
        med_counts = numpy.percentile(summed_counts, 50, axis=0)
        second_quar = numpy.percentile(summed_counts, 75, axis=0)
        nine_fifth = numpy.percentile(summed_counts, 95, axis=0)
        # compute median for comcat catalog
        obs_counts = []
        for mw in self.mws:
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            obs_binned_counts = numpy.zeros(self.n_bins)
            inds = bin1d_vec(obs_filt.get_epoch_times(), self.time_bins)
            for j in range(obs_filt.event_count):
                obs_binned_counts[inds[j]] += 1
            obs_counts.append(obs_binned_counts)
        obs_summed_counts = numpy.cumsum(obs_counts, axis=1)
        # update time_bins for plotting
        millis_to_hours = 60 * 60 * 1000 * 24
        time_bins = (self.time_bins - self.time_bins[0]) / millis_to_hours
        # since we are cumulating, plot at bin ends
        time_bins = time_bins + (self.dt / millis_to_hours)
        # make all arrays start at zero
        time_bins = numpy.insert(time_bins, 0, 0)
        # 2d array with (n_mw, n_time_bins)
        fifth_per = numpy.insert(fifth_per, 0, 0, axis=1)
        first_quar = numpy.insert(first_quar, 0, 0, axis=1)
        med_counts = numpy.insert(med_counts, 0, 0, axis=1)
        second_quar = numpy.insert(second_quar, 0, 0, axis=1)
        nine_fifth = numpy.insert(nine_fifth, 0, 0, axis=1)
        obs_summed_counts = numpy.insert(obs_summed_counts, 0, 0, axis=1)
        # ydata is now (5, n_mw, n_time_bins)
        results = {'xdata': time_bins,
                   'ydata': (fifth_per, first_quar, med_counts, second_quar, nine_fifth),
                   'obs_data': obs_summed_counts}

        return results

    def plot(self, results, plot_dir, show=False):
        # these are numpy arrays with mw information
        xdata = results['xdata']
        ydata = numpy.array(results['ydata'])
        obs_data = results['obs_data']
        # get values from plotting args
        for i, mw in enumerate(self.mws):
            cum_counts_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'cum_counts')
            plot_args = {'title': f'Cumulative Event Counts\nMw>{mw}',
                         'filename': cum_counts_fname}
            ax = plot_cumulative_events_versus_time_dev(xdata, ydata[:,i,:], obs_data[i,:], plot_args, show=False)
            # self.ax.append(ax)
            self.fnames.append(cum_counts_fname)

class MagnitudeHistogram(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc = calc

    def process_catalog(self, catalog):
        """ this can share data with the Magnitude test, hence self.calc
        """
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # always compute this for the lowest magnitude, above this is redundant
            cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
            binned_mags = cat_filt.binned_magnitude_counts()
            self.data.append(binned_mags)

    def evaluate(self, obs, args=None):
        """ just store observation for later """
        _ = args
        self.obs = obs

    def plot(self, results, plot_dir, show=False):
        mag_hist_fname = AbstractProcessingTask._build_figure_filename(plot_dir, self.mws[0], 'mag_hist')
        plot_args = {
            'xlim': [self.mws[0], numpy.max(CSEP_MW_BINS)],
             'title': f"Magnitude Histogram\nMw>{self.mws[0]}",
             'sim_label': self.name,
             'obs_label': self.obs.name,
             'filename': mag_hist_fname
        }
        obs_filt = self.obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        # data (n_sim, n_mag, n_mw_bins)
        ax = plot_magnitude_histogram_dev(numpy.array(self.data)[:,0,:], obs_filt, plot_args, show=False)
        # self.ax.append(ax)
        self.fnames.append(mag_hist_fname)

class InterEventTimeDistribution(AbstractProcessingTask):
    def process_catalog(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
        self.data.append(cat_filt.get_inter_event_times())

    def evaluate(self, obs, args=None):
        # get inter-event times from catalog
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        obs_terd = obs_filt.get_inter_event_times()

        # compute distribution statistics
        test_distribution, d_obs, quantile = _distribution_test(self.data, obs_terd)

        result = EvaluationResult(test_distribution=test_distribution,
                                  name='IEDD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  obs_catalog_repr=str(obs),
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, show=False):
        ietd_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, self.mws[0], 'ietd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                       'title': f'Inter-event Time Distribution Test\nMw>{self.mws[0]}',
                                                                       'bins': 'auto',
                                                                       'xlabel': "D* Statistic",
                                                                       'ylabel': r"P(X $\leq$ x)",
                                                                       'filename': ietd_test_fname})
        self.fnames.append(ietd_test_fname)

class InterEventDistanceDistribution(AbstractProcessingTask):
    def process_catalog(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
        self.data.append(cat_filt.get_inter_event_distances())

    def evaluate(self, obs, args=None):
        # get inter-event times from catalog
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        obs_terd = obs_filt.get_inter_event_distances()

        # compute distribution statistics
        test_distribution, d_obs, quantile = _distribution_test(self.data, obs_terd)

        result = EvaluationResult(test_distribution=test_distribution,
                                  name='IEDD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  obs_catalog_repr=str(obs),
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, show=False):
        iedd_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, self.mws[0], 'iedd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                       'title': f'Inter-event Distance Distribution Test\nMw>{self.mws[0]}',
                                                                       'bins': 'auto',
                                                                       'xlabel': "D* Statistic",
                                                                       'ylabel': r"P(X $\leq$ x)",
                                                                       'filename': iedd_test_fname})
        self.fnames.append(iedd_test_fname)

class TotalEventRateDistribution(AbstractProcessingTask):
    def __init__(self, cache=True, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self.mws = [2.5]

    def process_catalog(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        if self.region is None:
            self.region = catalog.region

        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
        # this is huge. could be 200k per catalog >>> n_events
        data = cat_filt.gridded_event_counts()
        if self.cache:
            if self.buffer_fname is None:
                self.buffer_fname = self._get_temporary_filename()
                self.fhandle = open(self.buffer_fname, 'wb+')
            self.cache_results(data)
        else:
            self.data.append(data)

    def evaluate(self, obs, args=None):
        # get inter-event times from catalog
        _, _, _, n_cat = args
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}')
        obs_terd = obs_filt.gridded_event_counts()

        if self.cache:
            # this might have to be buffered for large data, would be around 6Gb for 100k catalogs down to mw 2.5
            data = numpy.fromfile(self.buffer_fname).reshape(n_cat, len(self.mws), self.region.num_nodes)
        else:
            data = self.data

        # compute distribution statistics
        results = {}
        for i, mw in enumerate(self.mws):
            test_distribution, d_obs, quantile = _distribution_test(data[:,i,:], obs_terd)

            result = EvaluationResult(test_distribution=test_distribution,
                                  name='TERD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  obs_catalog_repr=str(obs),
                                  sim_name=self.name,
                                  obs_name=obs.name)
            results[mw] = result

        return results

    def plot(self, results, plot_dir, show=False):
        for mw, result in results.items():
            terd_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'terd_test')
            _ = plot_distribution_test(result, show=False, plot_args={'percentile': 95,
                                                                      'title': f'Total Event Rate Distribution-Test\nMw>{mw}',
                                                                      'bins': 'auto',
                                                                      'xlabel': "D* Statistic",
                                                                      'ylabel': r"P(X $\leq$ x)",
                                                                      'filename': terd_test_fname})
            self.fnames.append(terd_test_fname)

class BValueTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}', in_place=False)
        self.data.append(cat_filt.get_bvalue())

    def evaluate(self, obs, args=None):
        _ = args
        data = numpy.array(self.data)
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        observation_count = obs_filt.get_bvalue()
        # get delta_1 and delta_2 values
        _, delta_2 = get_quantiles(data, observation_count)
        # prepare result
        result = EvaluationResult(test_distribution=data,
                                  name='BV-Test',
                                  observed_statistic=observation_count,
                                  quantile=delta_2,
                                  status='Normal',
                                  obs_catalog_repr=str(obs),
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, show=False):
        bv_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, self.mws[0], 'bv_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"B-Value Distribution Test\nMw>{self.mws[0]}",
                                                             'bins': 'auto',
                                                             'xy': (0.6, 0.65),
                                                             'filename': bv_test_fname})
        self.fnames.append(bv_test_fname)

class MedianMagnitudeTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}', in_place=False)
        self.data.append(numpy.median(cat_filt.get_magnitudes()))
        print(self.data)

    def evaluate(self, obs, args=None):
        _ = args
        data = numpy.array(self.data)
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        observation_count = numpy.median(obs_filt.get_magnitudes())
        # get delta_1 and delta_2 values
        _, delta_2 = get_quantiles(data, observation_count)
        # prepare result
        result = EvaluationResult(test_distribution=data,
                                  name='MM-Test',
                                  observed_statistic=observation_count,
                                  quantile=delta_2,
                                  status='Normal',
                                  obs_catalog_repr=str(obs),
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, show=False):
        mm_test_fname = AbstractProcessingTask._build_figure_filename(plot_dir, self.mws[0], 'mm_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"Median Magnitude Distribution Test\nMw>{self.mws[0]}",
                                                             'bins': 25,
                                                             'filename': mm_test_fname})
        self.fnames.append(mm_test_fname)

class ConditionalEventsVersusTime(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_catalog(self, catalog):
        pass

    def evaluate(self, obs, args=None):
        pass

    def plot(self, results, plot_dir, show=False):
        pass

class SpatialLikelihoodPlot(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc = calc

    def process_catalog(self, catalog):
        # grab stuff from catalog that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # compute stuff from catalog
            counts = []
            for mw in self.mws:
                cat_filt = catalog.filter(f'magnitude > {mw}')
                counts.append(cat_filt.gridded_event_counts())
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(counts)
            else:
                self.data += numpy.array(counts)

    def evaluate(self, obs, args=None):
        self.obs = obs
        _, time_horizon, end_epoch, n_cat = args
        if len(self.data) == 0:
            raise ValueError("data is empty. need to have calc=True or manually bind data to the class.")
        data = numpy.array(self.data)
        apprx_rate_density = data / self.region.dh / self.region.dh / time_horizon / n_cat
        apprx_rate_density = apprx_rate_density / numpy.sum(apprx_rate_density, axis=1).reshape(-1,1)
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1) * self.region.dh * self.region.dh * time_horizon

        results = []
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            gridded_obs = obs_filt.gridded_event_counts()
            gridded_obs_ma = numpy.ma.masked_where(gridded_obs == 0, gridded_obs)
            apprx_rate_density_ma = numpy.ma.array(apprx_rate_density[i,:], mask=gridded_obs_ma.mask)
            likelihood = gridded_obs_ma * numpy.ma.log10(apprx_rate_density_ma) / obs_filt.event_count
            likelihood = likelihood.data
            result = self.region.get_cartesian(likelihood)
            results.append(result)
        return numpy.array(results)

    def plot(self, results, plot_dir, show=False):
        for i, mw in enumerate(self.mws):
            # compute expected rate density
            ax = plot_spatial_dataset(results[i,:,:],
                                      self.region,
                                      plot_args={'clabel': r'Pseudo Likelihood Per Event',
                                                 'clim': [-0.1, 0],
                                                 'title': f'Likelihood Plot with Observations\nMw > {mw}'})
            like_plot = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'like-plot')
            ax.figure.savefig(like_plot + '.pdf')
            ax.figure.savefig(like_plot + '.png')
            # self.ax.append(ax)
            self.fnames.append(like_plot)

class ConditionalRatePlot(AbstractProcessingTask):

    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc=calc
        self.region=None

    def process_catalog(self, catalog):
        # grab stuff from catalog that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # compute stuff from catalog
            counts = []
            for mw in self.mws:
                cat_filt = catalog.filter(f'magnitude > {mw}')
                counts.append(cat_filt.gridded_event_counts())
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(counts)
            else:
                self.data += numpy.array(counts)

    def evaluate(self, obs, args=None):
        """ store things for later """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return None

    def plot(self, results, plot_dir, show=False):
        crd = numpy.log10(numpy.array(self.data) / self.region.dh / self.region.dh / self.time_horizon / self.n_cat)

        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude > {mw}', in_place=False)
            plot_data = self.region.get_cartesian(crd[i,:])
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Conditional Rate Density'
                                                           '\n'
                                                           f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°)',
                                                 'clim': [0, 5],
                                                 'title': f'Approximate Rate Density with Observations\nMw > {mw}'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
            crd_fname = AbstractProcessingTask._build_figure_filename(plot_dir, mw, 'crd_obs')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
            # self.ax.append(ax)
            self.fnames.append(crd_fname)
