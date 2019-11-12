
import os
import json
import copy
from collections import defaultdict
from tempfile import mkstemp

import numpy

import datetime
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import time
from csep.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_WEEK, CSEP_MW_BINS
from csep import load_stochastic_event_sets, load_comcat
from csep.utils.time import epoch_time_to_utc_datetime, datetime_to_utc_epoch, millis_to_days
from csep.utils.spatial import masked_region, california_relm_region
from csep.utils.basic_types import Polygon, seq_iter
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.comcat import get_event_by_id
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR
from csep.utils.file import get_relative_path, mkdirs
from csep.utils.documents import MarkdownReport
from csep.core.evaluations import EvaluationResult, _compute_likelihood, _distribution_test
from csep.utils.plotting import plot_number_test, plot_magnitude_test, plot_likelihood_test, plot_spatial_test, \
    plot_cumulative_events_versus_time_dev, plot_magnitude_histogram_dev, plot_distribution_test, plot_spatial_dataset
from csep.utils.calc import bin1d_vec
from csep.utils.stats import get_quantiles, cumulative_square_diff
from csep.core.catalogs import ComcatCatalog
from csep.core.repositories import FileSystem



def ucerf3_consistency_testing(sim_dir, event_id, end_epoch, n_cat=None, plot_dir=None, generate_markdown=True, catalog_repo=None, save_results=False):
    """
    computes all csep consistency tests for simulation located in sim_dir with event_id

    Args:
        sim_dir (str): directory where results and configuration are stored
        event_id (str): event_id corresponding to comcat event
        data_products (dict):

    Returns:

    """
    # set up directories
    matplotlib.use('agg')
    matplotlib.rcParams['figure.max_open_warning'] = 150
    sns.set()

    # try using two different files
    filename = os.path.join(sim_dir, 'results_complete.bin')
    if not os.path.exists(filename):
        filename = os.path.join(sim_dir, 'results_complete_partial.bin')
    if not os.path.exists(filename):
        raise FileNotFoundError('could not find results_complete.bin or results_complete_partial.bin')
        
    if plot_dir is None:
        plot_dir = os.path.join(sim_dir, 'plots')
        print(f'No plotting directory specified defaulting to {plot_dir}')
    config_file = os.path.join(sim_dir, 'config.json')
    mkdirs(os.path.join(plot_dir))

    # load ucerf3 configuration
    with open(os.path.join(config_file), 'r') as f:
        u3etas_config = json.load(f)

    # determine how many catalogs to process
    if n_cat is None or n_cat > u3etas_config['numSimulations']:
        n_cat = u3etas_config['numSimulations']

    # download comcat information
    event = get_event_by_id(event_id)

    # filter to aftershock radius
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(event.magnitude) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((event.longitude, event.latitude),
                                                          3*rupture_length, num_points=100)
    aftershock_region = masked_region(california_relm_region(), aftershock_polygon)

    # event timing
    event_time = event.time.replace(tzinfo=datetime.timezone.utc)
    origin_epoch = u3etas_config['startTimeMillis']

    # this kinda booty
    if type(end_epoch) == str:
        print('Found end_epoch as time_delta string (in days), adding end_epoch to simulation start time')
        time_delta = 1000*24*60*60*int(end_epoch)
        end_epoch = origin_epoch + time_delta

    # convert epoch time (millis) to days
    time_horizon = (end_epoch - origin_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000

    # Download comcat catalog, if it fails its usually means it timed out, so just try again
    if catalog_repo is None:
        print("Catalog not specified downloading new catalog from ComCat.")

        # Sometimes ComCat fails for non-critical reasons, try twice just to make sure.
        try:
            comcat = load_comcat(epoch_time_to_utc_datetime(origin_epoch), epoch_time_to_utc_datetime(end_epoch),
                                      min_magnitude=2.50,
                                      min_latitude=31.50, max_latitude=43.00,
                                      min_longitude=-125.40, max_longitude=-113.10)
            comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, origin_epoch)
            print(comcat)
        except:
            comcat = load_comcat(event_time, epoch_time_to_utc_datetime(end_epoch),
                                      min_magnitude=2.50,
                                      min_latitude=31.50, max_latitude=43.00,
                                      min_longitude=-125.40, max_longitude=-113.10)
            comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, origin_epoch)
            print(comcat)
    else:
        # if this fails it should stop the program, therefore no try-catch block
        catalog_repo = FileSystem(url=catalog_repo)

        print(f"Reading catalog from repository at location {catalog_repo.url}")
        comcat = catalog_repo.load(ComcatCatalog(query=False))
        comcat = comcat.filter(f'origin_time >= {datetime_to_utc_epoch(event_time)}').filter(f'origin_time < {end_epoch}')
        comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, origin_epoch)


    # define products to compute on simulation, this could be extracted
    data_products = {
         'n-test': NumberTest(),
         'm-test': MagnitudeTest(),
         'l-test': LikelihoodAndSpatialTest(),
         'cum-plot': CumulativeEventPlot(origin_epoch, end_epoch),
         'mag-hist': MagnitudeHistogram(calc=False),
         'arp-plot': ApproximateRatePlot(calc=False),
         'carp-plot': ConditionalApproximateRatePlot(comcat),
         'bv-test': BValueTest()
    }

    print(f'Will process {n_cat} catalogs from simulation\n')
    for k, v in data_products.items():
        print(f'Computing {v.__class__.__name__}')
    print('\n')

    # read the catalogs
    print('Begin processing catalogs')
    t0 = time.time()
    loaded = 0
    u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name='UCERF3-ETAS', region=aftershock_region)
    try:
        for i, cat in enumerate(u3):
            cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region).apply_mct(event.magnitude, origin_epoch)
            for name, calc in data_products.items():
                calc.process_catalog(copy.copy(cat_filt))
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i+1} catalogs in {t1-t0} seconds')
            if (i+1) % n_cat == 0:
                break
            loaded += 1
    except:
        print(f'Failed loading at catalog {i+1}. This may happen if the simulation is incomplete\nProceeding to finalize plots')
        n_cat = loaded

    t2 = time.time()
    print(f'Finished processing catalogs in {t2-t0} seconds\n')

    print('Processing catalogs again for distribution-based tests')
    for k, v in data_products.items():
        if v.needs_two_passes == True:
            print(v.__class__.__name__)
    print('\n')

    # share data where applicable
    data_products['mag-hist'].data = data_products['m-test'].data
    data_products['arp-plot'].data = data_products['l-test'].data

    # old iterator is expired, need new one
    t2 = time.time()
    u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name='UCERF3-ETAS', region=aftershock_region)
    for i, cat in enumerate(u3):
        cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region).apply_mct(event.magnitude, origin_epoch)
        for name, calc in data_products.items():
            calc.process_again(copy.copy(cat_filt), args=(time_horizon, n_cat, end_epoch, comcat))
        # if we failed earlier, just stop there again
        tens_exp = numpy.floor(numpy.log10(i+1))
        if (i+1) % 10**tens_exp == 0:
            t3 = time.time()
            print(f'Processed {i + 1} catalogs in {t3 - t2} seconds')
        if (i+1) % n_cat == 0:
            break

    # evaluate the catalogs and store results
    t1 = time.time()

    # make plot directory
    fig_dir = os.path.join(plot_dir, 'plots')
    mkdirs(fig_dir)

    # make results directory
    if save_results:
        results_dir = os.path.join(plot_dir, 'results')
        mkdirs(results_dir)

    for name, calc in data_products.items():
        print(f'Finalizing calculations for {name} and plotting')
        result = calc.post_process(comcat, args=(u3, time_horizon, end_epoch, n_cat))
        # plot, and store in plot_dir
        calc.plot(result, fig_dir, show=False)

        if save_results:
            # could expose this, but hard-coded for now
            print(f"Storing results from evaluations in {results_dir}")
            calc.store_results(result, results_dir)

    t2 = time.time()
    print(f"Evaluated forecasts in {t2-t1} seconds")

    # writing catalog
    print(f"Saving ComCat catalog used for Evaluation")
    evaluation_repo = FileSystem(url=os.path.join(plot_dir, 'evaluation_catalog.json'))
    evaluation_repo.save(comcat.to_dict())

    print(f"Finished everything in {t2-t0} seconds with average time per catalog of {(t2-t0)/n_cat} seconds")

    # create the notebook for results
    if generate_markdown:
        md = MarkdownReport('README.md')

        md.add_introduction(adict={'simulation_name': u3etas_config['simulationName'],
                                   'origin_time': epoch_time_to_utc_datetime(origin_epoch),
                                   'evaluation_time': epoch_time_to_utc_datetime(end_epoch),
                                   'catalog_source': 'ComCat',
                                   'forecast_name': 'UCERF3-ETAS',
                                   'num_simulations': n_cat})

        md.add_sub_heading('Visual Overview of Forecast', 1,
                "These plots show qualitative comparisons between the forecast "
                f"and the target catalog obtained from ComCat. Plots contain events within {numpy.round(millis_to_days(end_epoch-origin_epoch))} days "
                f"of the forecast start time and within {numpy.round(3*rupture_length/1000)} kilometers from the epicenter of the mainshock.  \n  \n"
                "All catalogs are processed using a time-dependent magnitude of completeness from Helmstetter et al., 2006.\n")

        md.add_result_figure('Cumulative Event Counts', 2, list(map(get_relative_path, data_products['cum-plot'].fnames)), ncols=2,
                             text="Percentiles for cumulative event counts are aggregated within one-day bins.  \n")

        md.add_result_figure('Magnitude Histogram', 2, list(map(get_relative_path, data_products['mag-hist'].fnames)))

        md.add_result_figure('Approximate Rate Density with Observations', 2, list(map(get_relative_path, data_products['arp-plot'].fnames)), ncols=2)

        md.add_result_figure('Conditional Rate Density', 2, list(map(get_relative_path, data_products['carp-plot'].fnames)), ncols=2,
                             text="Plots are conditioned on number of target events ± 5%\n")

        md.add_sub_heading('CSEP Consistency Tests', 1, "<b>Note</b>: These tests are explained in detail by Savran et al. (In prep).\n")

        md.add_result_figure('Number Test', 2, list(map(get_relative_path, data_products['n-test'].fnames)),
                             text="The number test compares the earthquake counts within the forecast region aginst observations from the"
                                  " target catalog.\n")

        md.add_result_figure('Magnitude Test', 2, list(map(get_relative_path, data_products['m-test'].fnames)),
                             text="The magnitude test computes the sum of squared residuals between normalized "
                                  "incremental Magnitude-Number distributions."
                                  " The test distribution is built from statistics scored between individal catalogs and the"
                                  " expected Magnitude-Number distribution of the forecast.\n")

        md.add_result_figure('Likelihood Test', 2, list(map(get_relative_path, data_products['l-test'].fnames['l-test'])),
                             text="The likelihood tests uses a statistic based on the continuous point-process "
                                  "likelihood function. We approximate the rate-density of the forecast "
                                  "by stacking synthetic catalogs in spatial bins. The rate-density represents the "
                                  "probability of observing an event selected at random from the forecast. "
                                  "Event log-likelihoods are aggregated for each event in the catalog. This "
                                  "approximation to the continuous rate-density is unconditional in the sense that it does "
                                  "not consider the number of target events.\n")

        md.add_result_figure('Spatial Test', 2, list(map(get_relative_path, data_products['l-test'].fnames['s-test'])),
                             text="The spatial test is based on the same likelihood statistic from above. However, "
                                  "the scores are normalized so that differences in earthquake rates are inconsequential. "
                                  "As above, this statistic is unconditional.\n")


        md.add_sub_heading('One-point Statistics', 1, "")
        md.add_result_figure('B-Value Test', 2, list(map(get_relative_path, data_products['bv-test'].fnames)),
                             text="This test compares the estimated b-value from the observed catalog along with the "
                                  "b-value distribution from the forecast. "
                                  "This test can be considered an alternate form to the Magnitude Test.\n")
        md.finalize(os.path.dirname(plot_dir))

    t1 = time.time()
    print(f'Completed all processing in {t1-t0} seconds')


class AbstractProcessingTask:
    def __init__(self, data=None, name=None):
        self.data = data or []
        self.mws = [2.5, 3.0, 3.5, 4.0, 4.5]
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
    def _build_filename(dir, mw, plot_id):
        basename = f"{plot_id}_{mw}_test"
        return os.path.join(dir, basename)

    @staticmethod
    def _get_temporary_filename():
        # create temporary file and return filename
        _, tmp_file = mkstemp()
        return tmp_file

    def process_catalog(self, catalog):
        raise NotImplementedError('must implement process_catalog()!')

    def process_again(self, catalog, args=()):
        """ This function defaults to pass unless the method needs to read through the data twice. """
        pass

    def post_process(self, obs, args=None):
        """
        Compute evaluation of data stored in self.data.

        Args:
            obs (csep.Catalog): used to evaluate the forecast
            args (tuple): args for this function

        Returns:
            result (csep.core.evaluations.EvaluationResult):

        """
        result = EvaluationResult()
        return result

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

    def store_results(self, results, dir):
        """
        Saves evaluation results serialized into json format. This format is used to recreate the results class which
        can then be plotted if desired. The following directory structure will be created:

        | dir
        |-- n-test
        |---- n-test_mw_2.5.json
        |---- n_test_mw_3.0.json
        |-- m-test
        |---- m_test_mw_2.5.json
        |---- m_test_mw_3.0.json
        ...

        The results iterable should only contain results for a single evaluation. Typically they would contain different
        minimum magnitudes.

        Args:
            results (Iterable of EvaluationResult): iterable object containing evaluation results. this could be a list or tuple of lists as well
            dir (str): directory to store the testing results. name will be constructed programatically.

        Returns:
            None

        """
        success = False
        for idx in seq_iter(results):
            if not isinstance(results[idx], tuple) or not isinstance(results[idx], list):
                result = [results[idx]]
            else:
                result = results[idx]
            for r in result:
                repo = FileSystem(url=self._build_filename(dir, r.min_mw, r.name).lower() + '.json')
                if repo.save(r.to_dict()):
                    success = True
        return success


class NumberTest(AbstractProcessingTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            counts.append(cat_filt.event_count)
        self.data.append(counts)

    def post_process(self, obs, args=None):
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
                          obs_catalog_repr=obs.date_accessed,
                          sim_name=self.name,
                          min_mw=mw,
                          obs_name=obs.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):

        for mw, result in results.items():
            # compute bin counts, this one is special because of integer values
            td = result.test_distribution
            min_bin, max_bin = numpy.min(td), numpy.max(td)
            # hard-code some logic for bin size
            bins = numpy.arange(min_bin, max_bin)
            n_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'n_test')
            _ = plot_number_test(result, show=show, plot_args={'percentile': 95,
                                                                'title': f'Number-Test\nMw>{mw}',
                                                                'bins': bins,
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

    def post_process(self, obs, args=None):
        # we dont need args
        _ = args
        results = {}
        for i, mw in enumerate(self.mws):
            test_distribution = []
            # get observed magnitude counts
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            if obs_filt.event_count == 0:
                continue
            obs_histogram = obs_filt.binned_magnitude_counts()
            n_obs_events = numpy.sum(obs_histogram)
            mag_counts_all = numpy.array(self.data)
            # get the union histogram, simply the sum over all catalogs, (n_cat, n_mw)
            union_histogram = numpy.sum(mag_counts_all[:,i,:], axis=0)
            n_union_events = numpy.sum(union_histogram)
            union_scale = n_obs_events / n_union_events
            scaled_union_histogram = union_histogram * union_scale
            for j in range(mag_counts_all.shape[0]):
                n_events = numpy.sum(mag_counts_all[j,i,:])
                if n_events == 0:
                    continue
                scale = n_obs_events / n_events
                catalog_histogram = mag_counts_all[j,i,:] * scale

                test_distribution.append(cumulative_square_diff(catalog_histogram, scaled_union_histogram))
            # compute statistic from the observation
            obs_d_statistic = cumulative_square_diff(obs_histogram, scaled_union_histogram)
            # score evaluation
            _, quantile = get_quantiles(test_distribution, obs_d_statistic)
            # prepare result
            result = EvaluationResult(test_distribution=test_distribution,
                                      name='M-Test',
                                      observed_statistic=obs_d_statistic,
                                      quantile=quantile,
                                      status='Normal',
                                      min_mw=mw,
                                      obs_catalog_repr=obs.date_accessed,
                                      obs_name=obs.name,
                                      sim_name=self.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # get the filename
        for mw, result in results.items():
            m_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'm-test')
            plot_args = {'percentile': 95,
                         'title': f'Magnitude-Test\nMw>{mw}',
                         'bins': 'auto',
                         'filename': m_test_fname}
            _ = plot_magnitude_test(result, show=False, plot_args=plot_args)
            self.fnames.append(m_test_fname)


class LikelihoodAndSpatialTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution_spatial = []
        self.test_distribution_likelihood = []
        self.cat_id = 0
        self.needs_two_passes = True
        self.buffer = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []

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
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        time_horizon, n_cat, end_epoch, obs = args
        apprx_rate_density = self.data / self.region.dh / self.region.dh / time_horizon / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1) * self.region.dh * self.region.dh * time_horizon

        # unfortunately, we need to iterate twice through the catalogs for this.
        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude > {mw}')
            gridded_cat = cat_filt.gridded_event_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm
        self.test_distribution_likelihood.append(lhs)
        self.test_distribution_spatial.append(lhs_norm)

    def post_process(self, obs, args=None):
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}

        apprx_rate_density = self.data / self.region.dh / self.region.dh / time_horizon / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1) * self.region.dh * self.region.dh * time_horizon

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
            message = "normal"
            # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
            # either normal and wrong or udetermined (undersampled)
            if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh):
                    # Build message
                    message = "undetermined"
            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:,i],
                                                 name='L-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)
            # find out if there are issues with the test
            if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh_norm):
                    # Build message
                    message = "undetermined"

            if n_obs == 0:
                message = 'not-valid'

            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial[:,i],
                                          name='S-Test',
                                          observed_statistic=obs_lh_norm,
                                          quantile=quantile_spatial,
                                          status=message,
                                          min_mw=mw,
                                          obs_catalog_repr=obs.date_accessed,
                                          sim_name=self.name,
                                          obs_name=obs.name)

            results[mw] = (result_likelihood, result_spatial)

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result_tuple in results.items():
            # plot likelihood test
            l_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'l-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test\nMw>{mw}',
                         'bins': 'auto',
                         'filename': l_test_fname}
            _ = plot_likelihood_test(result_tuple[0], axes=None, plot_args=plot_args, show=show)

            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['l-test'].append(l_test_fname)

            if result_tuple[1].status == 'not-valid':
                print(f'Skipping plot for spatial test on {mw}. Test results are not valid, likely because no earthquakes observed in target catalog.')
                continue

            # plot spatial test
            s_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 's-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test\nMw>{mw}',
                         'bins': 'auto',
                         'filename': s_test_fname}
            _ = plot_spatial_test(result_tuple[1], axes=None, plot_args=plot_args, show=False)
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

    def post_process(self, obs, args=None):
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

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # these are numpy arrays with mw information
        xdata = results['xdata']
        ydata = numpy.array(results['ydata'])
        obs_data = results['obs_data']
        # get values from plotting args
        for i, mw in enumerate(self.mws):
            cum_counts_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'cum_counts')
            plot_args = {'title': f'Cumulative Event Counts\nMw>{mw}',
                         'xlabel': 'Days Since Start of Forecast',
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

    def post_process(self, obs, args=None):
        """ just store observation for later """
        _ = args
        self.obs = obs

    def plot(self, results, plot_dir, plot_args=None, show=False):
        mag_hist_fname = AbstractProcessingTask._build_filename(plot_dir, self.mws[0], 'mag_hist')
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


class UniformLikelihoodCalculation(AbstractProcessingTask):
    """
    This calculation assumes that the spatial distribution of the forecast is uniform, but the seismicity is located
    in spatial bins according to the clustering provided by the forecast model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.test_distribution_likelihood = []
        self.test_distribution_spatial = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []
        self.needs_two_passes = True

    def process_catalog(self, catalog):
        # grab stuff from catalog that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name

    def process_again(self, catalog, args=()):

        time_horizon, n_cat, end_epoch, obs = args

        expected_cond_count = numpy.sum(self.data, axis=1) / n_cat

        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))

        for i, mw in enumerate(self.mws):

            # generate with uniform rate in every spatial bin
            apprx_rate_density = expected_cond_count[i] * numpy.ones(self.region.num_nodes) / self.region.num_nodes

            # convert to rate density
            apprx_rate_density = apprx_rate_density / self.region.dh / self.region.dh / time_horizon

            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude > {mw}')
            gridded_cat = cat_filt.gridded_event_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density, expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm

        self.test_distribution_likelihood.append(lhs)
        self.test_distribution_spatial.append(lhs_norm)

    def post_process(self, obs, args=None):

        _, time_horizon, _, n_cat = args

        results = {}
        expected_cond_count = numpy.sum(self.data, axis=1) / n_cat

        test_distribution_likelihood = numpy.array(self.test_distribution_likelihood)
        test_distribution_spatial = numpy.array(self.test_distribution_spatial)

        for i, mw in enumerate(self.mws):

            # create uniform apprx rate density
            apprx_rate_density = expected_cond_count[i] * numpy.ones(self.region.num_nodes) / self.region.num_nodes

            # convert to rate density
            apprx_rate_density = apprx_rate_density / self.region.dh / self.region.dh / time_horizon

            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.gridded_event_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density, expected_cond_count[i],
                                                      n_obs)
            # determine outcome of evaluation, check for infinity
            _, quantile_likelihood = get_quantiles(test_distribution_likelihood[:, i], obs_lh)
            _, quantile_spatial = get_quantiles(test_distribution_spatial[:, i], obs_lh_norm)

            # Signals outcome of test
            message = "normal"
            # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
            # either normal and wrong or udetermined (undersampled)
            if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh):
                    # Build message
                    message = "undetermined"

            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:, i],
                                                 name='UL-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)
            # find out if there are issues with the test
            if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh_norm):
                    # Build message
                    message = "undetermined"

            if n_obs == 0:
                message = 'not-valid'

            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial[:, i],
                                              name='US-Test',
                                              observed_statistic=obs_lh_norm,
                                              quantile=quantile_spatial,
                                              status=message,
                                              min_mw=mw,
                                              obs_catalog_repr=obs.date_accessed,
                                              sim_name=self.name,
                                              obs_name=obs.name)

            results[mw] = (result_likelihood, result_spatial)

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result_tuple in results.items():
            # plot likelihood test
            l_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'ul-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test\nMw>{mw}',
                         'bins': 'fd',
                         'filename': l_test_fname}
            _ = plot_likelihood_test(result_tuple[0], axes=None, plot_args=plot_args, show=show)

            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['l-test'].append(l_test_fname)

            if result_tuple[1].status == 'not-valid':
                print(
                    f'Skipping plot for spatial test on {mw}. Test results are not valid, likely because no earthquakes observed in target catalog.')
                continue

            # plot spatial test
            s_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'us-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test\nMw>{mw}',
                         'bins': 'fd',
                         'filename': s_test_fname}
            _ = plot_spatial_test(result_tuple[1], axes=None, plot_args=plot_args, show=False)
            self.fnames['s-test'].append(s_test_fname)


class InterEventTimeDistribution(AbstractProcessingTask):
    def process_catalog(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
        self.data.append(cat_filt.get_inter_event_times())

    def post_process(self, obs, args=None):
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
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def process_again(self, catalog, args=()):
        pass

    def plot(self, results, plot_dir, plot_args=None, show=False):
        ietd_test_fname = AbstractProcessingTask._build_filename(plot_dir, self.mws[0], 'ietd_test')
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

    def post_process(self, obs, args=None):
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
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        iedd_test_fname = AbstractProcessingTask._build_filename(plot_dir, self.mws[0], 'iedd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                       'title': f'Inter-event Distance Distribution Test\nMw>{self.mws[0]}',
                                                                       'bins': 'auto',
                                                                       'xlabel': "D* Statistic",
                                                                       'ylabel': r"P(X $\leq$ x)",
                                                                       'filename': iedd_test_fname})
        self.fnames.append(iedd_test_fname)


class TotalEventRateDistribution(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5]

    def process_catalog(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        if self.region is None:
            self.region = catalog.region

        cat_filt = catalog.filter(f'magnitude > {self.mws[0]}')
        data = cat_filt.gridded_event_counts()
        self.data.append(data)

    def post_process(self, obs, args=None):
        # get inter-event times from catalog
        _, _, _, n_cat = args
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        obs_terd = obs_filt.gridded_event_counts()
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
                                  min_mw=mw,
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)
            results[mw] = result

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result in results.items():
            terd_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'terd_test')
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

    def post_process(self, obs, args=None):
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
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        bv_test_fname = AbstractProcessingTask._build_filename(plot_dir, self.mws[0], 'bv_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"B-Value Distribution Test\nMw>{self.mws[0]}",
                                                             'bins': 'auto',
                                                             'xy': (0.2, 0.65),
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

    def post_process(self, obs, args=None):
        _ = args
        data = numpy.array(self.data)
        obs_filt = obs.filter(f'magnitude > {self.mws[0]}', in_place=False)
        observation_count = float(numpy.median(obs_filt.get_magnitudes()))
        # get delta_1 and delta_2 values
        _, delta_2 = get_quantiles(data, observation_count)
        # prepare result
        result = EvaluationResult(test_distribution=data,
                                  name='M-Test',
                                  observed_statistic=observation_count,
                                  quantile=delta_2,
                                  min_mw=self.mws[0],
                                  status='Normal',
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        mm_test_fname = AbstractProcessingTask._build_filename(plot_dir, self.mws[0], 'mm_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"Median Magnitude Distribution Test\nMw>{self.mws[0]}",
                                                             'bins': 25,
                                                             'filename': mm_test_fname})
        self.fnames.append(mm_test_fname)


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

    def post_process(self, obs, args=None):
        self.obs = obs
        _, time_horizon, end_epoch, n_cat = args
        if len(self.data) == 0:
            raise ValueError("data is empty. need to have calc=True or manually bind data to the class.")
        data = numpy.array(self.data)
        apprx_rate_density = data / self.region.dh / self.region.dh / time_horizon / n_cat
        apprx_rate_density = apprx_rate_density / numpy.sum(apprx_rate_density, axis=1).reshape(-1,1)

        results = []
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude > {mw}', in_place=False)
            if obs_filt.event_count == 0:
                continue
            gridded_obs = obs_filt.gridded_event_counts()
            gridded_obs_ma = numpy.ma.masked_where(gridded_obs == 0, gridded_obs)
            apprx_rate_density_ma = numpy.ma.array(apprx_rate_density[i,:], mask=gridded_obs_ma.mask)
            likelihood = gridded_obs_ma * numpy.ma.log10(apprx_rate_density_ma) / obs_filt.event_count
            likelihood = likelihood.data
            result = self.region.get_cartesian(likelihood)
            results.append(result)
        return numpy.array(results)

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for i, mw in enumerate(self.mws):
            # compute expected rate density
            try:
                ax = plot_spatial_dataset(results[i,:,:],
                                          self.region,
                                          plot_args={'clabel': r'Pseudo Likelihood Per Event',
                                                     'clim': [-0.1, 0],
                                                     'title': f'Likelihood Plot with Observations\nMw > {mw}'})
                like_plot = AbstractProcessingTask._build_filename(plot_dir, mw, 'like-plot')
                ax.figure.savefig(like_plot + '.pdf')
                ax.figure.savefig(like_plot + '.png')
                # self.ax.append(ax)
                self.fnames.append(like_plot)
            except IndexError:
                print(f'Skipping plotting of Mw: {mw}, results not found for this magnitude')


class ApproximateRatePlot(AbstractProcessingTask):

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
                gridded_counts = cat_filt.gridded_event_counts()
                counts.append(gridded_counts)
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(counts)
            else:
                self.data += numpy.array(counts)

    def post_process(self, obs, args=None):
        """ store things for later """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return None

    def plot(self, results, plot_dir, plot_args=None, show=False):
        crd = numpy.log10(numpy.array(self.data) / self.region.dh / self.region.dh / self.time_horizon / self.n_cat)

        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude > {mw}', in_place=False)
            plot_data = self.region.get_cartesian(crd[i,:])
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Approximate Rate Density'
                                                           '\n'
                                                           f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°)',
                                                 'clim': [0, 5],
                                                 'title': f'Approximate Rate Density with Observations\nMw > {mw}'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
            crd_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'crd_obs')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
            # self.ax.append(ax)
            self.fnames.append(crd_fname)


class ConditionalApproximateRatePlot(AbstractProcessingTask):

    def __init__(self, obs, **kwargs):
        super().__init__(**kwargs)
        self.obs = obs
        self.data = defaultdict(list)

    def process_catalog(self, catalog):
        if self.name is None:
            self.name = catalog.name

        if self.region is None:
            self.region = catalog.region
        """ collects all catalogs conforming to n_obs in a dict"""
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            obs_filt = self.obs.filter(f'magnitude > {mw}', in_place=False)
            n_obs = obs_filt.event_count
            tolerance = 0.05 * n_obs
            if cat_filt.event_count <= n_obs + tolerance \
                and cat_filt.event_count >= n_obs - tolerance:
                self.data[mw].append(cat_filt.gridded_event_counts())

    def post_process(self, obs, args=None):
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # compute conditional approximate rate density
        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude > {mw}', in_place=False)
            if obs_filt.event_count == 0:
                continue

            rates = numpy.array(self.data[mw])
            if rates.shape[0] == 0:
                continue

            # compute conditional approximate rate
            mean_rates = numpy.mean(rates, axis=0)
            crd = numpy.log10(mean_rates / self.region.dh / self.region.dh / self.time_horizon)
            plot_data = self.region.get_cartesian(crd)
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Conditional Rate Density'
                                                           '\n'
                                      f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°)',
                                                 'clim': [0, 5],
                                                 'title': f'Conditional Approximate Rate Density with Observations\nMw > {mw}'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40,
                       edgecolors='black')
            crd_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'cond_rates')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
                # self.ax.append(ax)
            self.fnames.append(crd_fname)


class ConditionalMagnitudeVersusTime(AbstractProcessingTask):

    def __init__(self, obs, data=None, name=None):
        super().__init__(data, name)

        # we try a few different magnitudes becuase they are not guaranteed to work
        self.mws = [2.5, 3.0, 3.5]
        self.n_obs = []
        self.saved_catalog = None

        for mw in self.mws:
            obs_filt = obs.filter(f'magnitude > {mw}')
            self.n_obs.append(obs_filt.event_count)

    def process_catalog(self, catalog):

        for i, mw in enumerate(self.mws):
            cat_filt = catalog.filter(f'magntiude > {mw}')
            n_cat = cat_filt.event_count


class CatalogMeanStabilityAnalysis(AbstractProcessingTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calc = False
        self.mws = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

    def process_catalog(self, catalog):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude > {mw}')
            counts.append(cat_filt.event_count)
        self.data.append(counts)

    def post_process(self, obs, args=None):
        results = {}
        data = numpy.array(self.data)
        n_sim = data.shape[0]
        end_points = numpy.arange(1,n_sim,100)
        for i, mw in enumerate(self.mws):
            running_means = []
            for ep in end_points:
                running_means.append(numpy.mean(data[:ep,i]))
            results[mw] = (end_points, running_means)
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw in self.mws:
            fname = self._build_filename(plot_dir, mw, 'comp_test')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            res = numpy.array(results[mw])
            ax.plot(res[0,:], res[1,:])
            ax.set_title(f'Catalog Mean Stability Mw > {mw}')
            ax.set_xlabel('Average Event Count')
            ax.set_ylabel('Running Mean')
            fig.savefig(fname + '.png')
            plt.close(fig)