# native imports
import functools
from collections import defaultdict

import time
import datetime
import os
import multiprocessing

# 3rd party imports
import tqdm
import numpy as np

# CSEP Imports
import csep
from csep.core.evaluations import number_test, magnitude_test, interevent_time_test, interevent_distance_test, \
    bvalue_test
from csep.core.evaluations import combined_likelihood_and_spatial
from csep.utils.plotting import plot_magnitude_histogram, plot_distribution_test
from csep.utils.time import datetime_to_utc_epoch, epoch_time_to_utc_datetime
from csep.utils.spatial import california_relm_region, masked_region
from csep.utils.plotting import plot_number_test, plot_magnitude_test, plot_likelihood_test, plot_spatial_test
from csep.utils.plotting import plot_spatial_dataset, plot_cumulative_events_versus_time
from csep.utils.constants import CSEP_MW_BINS, SECONDS_PER_ASTRONOMICAL_YEAR, MW_5_EQS_PER_YEAR
from csep.utils.comcat import get_event_by_id
from csep.utils.documents import ResultsNotebook
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.basic_types import Polygon
from csep.utils.file import mkdirs, get_relative_path

def main(mw_min, config, sim_dir, end_time):
    # set things up
    filename = os.path.join(sim_dir, 'results_complete.bin')
    relm_region = california_relm_region()
    end_epoch = csep.utils.time.datetime_to_utc_epoch(end_time)
    plot_dir = os.path.join(sim_dir, 'plotting')
    mkdirs(plot_dir)

    # time-horizon of forecast, epochs are in milliseconds
    origin_time = config['startTimeMillis']
    time_horizon = (end_epoch - origin_time) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000
    n_cat = config['numSimulations']
    e_lon, e_lat = config['eventLongitude'], config['eventLatitude']

    # build wells and coppersmith radius in meters, based on hypocenter not fault trace
    t0 = time.time()
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(config['eventMagnitude']) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((e_lon, e_lat), 3*rupture_length, num_points=100)
    aftershock_region = masked_region(relm_region, aftershock_polygon)
    t1 = time.time()
    print(f'Created new masked aftershock region in {t1-t0} seconds.')

    # Download comcat catalog
    print('Loading Comcat.')
    t0 = time.time()
    comcat = csep.core.catalogs.ComcatCatalog(start_epoch=origin_time, end_time=end_time,
                                              name='Comcat', min_magnitude=2.50,
                                              min_latitude=31.50, max_latitude=43.00,
                                              min_longitude=-125.40, max_longitude=-113.10, region=relm_region, query=True)
    t1 = time.time()
    print("Fetched Comcat catalog in {} seconds.\n".format(t1 - t0))
    print("Downloaded Comcat Catalog with following parameters")
    print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
    print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
    print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
    print("Min Magnitude: {}".format(comcat.min_magnitude))
    print(f"Found {comcat.get_number_of_events()} events in the Comcat catalog.")
    print(f'Proceesing Catalogs.')

    comcat = comcat.filter(f'magnitude >= {mw_min}').filter_spatial(aftershock_region)
    print('Filtered Comcat catalog.')
    print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
    print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
    print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
    print("Min Magnitude: {}".format(comcat.min_magnitude))
    print(f"Found {comcat.get_number_of_events()} events in the Comcat catalog.")

    # used to communicate results of evaluations to other processes
    results = {}
    results['catalog_source'] = comcat.name
    results['minimum_mw'] = mw_min

    # Process Stochastic Event Sets
    gridded_counts = np.zeros(aftershock_region.num_nodes)
    u3_filt = []

    u3catalogs = csep.load_stochastic_event_set(filename=filename, type='ucerf3', format='native', name='UCERF3-ETAS', region=aftershock_region)
    # read catalogs from disk
    for i in tqdm.tqdm(range(n_cat), total=n_cat):
        cat = next(u3catalogs)
        cat_filt = cat.filter(f'origin_time < {end_epoch}').filter(f'magnitude >= {mw_min}').filter_spatial(aftershock_region)
        # store catalog will need to iterate over these again to compute pseudo-likelihood
        u3_filt.append(cat_filt)
        # counts on spatial grid, might as well do now
        gridded_counts += cat_filt.gridded_event_counts()
    n_cat = len(u3_filt)


    print("Computing N-test results.")
    t0 = time.time()
    n_test_result = number_test(u3_filt, comcat)
    n_test_fname = build_figure_filename(plot_dir, mw_min, 'n_test')
    _ = plot_number_test(n_test_result, show=False,
                         plot_args={'percentile': 95,
                                    'title': f'Number-Test\nMw>{mw_min}',
                                    'bins': 'auto',
                                    'filename': n_test_fname})
    # store results for later, we care about the test result tuple and the test filename


    cum_counts_fname = build_figure_filename(plot_dir, mw_min, 'cum_counts')
    _ = plot_cumulative_events_versus_time(u3_filt, comcat, show=False,
                                           plot_args={'title': f'Cumulative Event Counts\nMw>{mw_min}',
                                                      'filename': cum_counts_fname})

    t1 = time.time()
    print(f'Done with N-test in {t1-t0} seconds.')

    print("Computing M-test results.")
    t0 = time.time()
    m_test_result = magnitude_test(u3_filt, comcat)
    m_test_fname = build_figure_filename(plot_dir, mw_min, 'm_test')
    _ = plot_magnitude_test(m_test_result, show=False, plot_args={'percentile': 95,
                                                                  'title': f'Magnitude-Test\nMw>{mw_min}',
                                                                  'bins': 'auto',
                                                                  'filename': m_test_fname})

    print("Plotting Magnitude Histogram")
    mag_hist_fname = build_figure_filename(plot_dir, mw_min, 'mag_hist')
    _ = plot_magnitude_histogram(u3_filt, comcat, show=False, plot_args={'xlim': [mw_min, np.max(CSEP_MW_BINS)],
                                                                         'title': f"Magnitude Histogram\nMw>{mw_min}",
                                                                         'sim_label': u3_filt[0].name,
                                                                         'filename':  mag_hist_fname})

    t1 = time.time()
    print(f'Done with M-test in {t1-t0} seconds.')

    print("Computing S-test and L-test results.")
    t0 = time.time()
    cond_rate_dens = gridded_counts/n_cat/aftershock_region.dh/aftershock_region.dh/time_horizon
    l_test_result, s_test_result = combined_likelihood_and_spatial(u3_filt, comcat, cond_rate_dens, time_horizon)
    print(f"{l_test_result.name} result is {l_test_result.status}")
    l_test_fname = build_figure_filename(plot_dir, mw_min, 'l_test')
    _ = plot_likelihood_test(l_test_result, show=False,
                              plot_args={'percentile': 95,
                                         'title': f'Pseduo Likelihood-Test\nMw>{mw_min}',
                                         'bins': 'auto',
                                         'filename':  l_test_fname})

    t1 = time.time()
    print(f'Done with S-test and L-test in {t1-t0} seconds.')

    t0 = time.time()
    print(f"{s_test_result.name} result is {s_test_result.status}")
    s_test_fname = build_figure_filename(plot_dir, mw_min, 's_test')
    _ = plot_spatial_test(s_test_result, show=False,
                           plot_args={'percentile': 95,
                                      'title': f'Spatial-Test\nMw>{mw_min}',
                                      'bins': 'auto',
                                      'filename':s_test_fname})

    print("Plotting spatial distribution of events.")
    # used for plotting, converted to 2d grid
    log_cond_rate_dens_cart = aftershock_region.get_cartesian(np.log10(cond_rate_dens))
    log_cond_rate_dens_cart[log_cond_rate_dens_cart == -np.inf] = -9999
    # compute color bar given rate of certain magnitude events, hard-coded for CSEP MW Bins, assuming 10 Mw 5 Per year
    vmax = np.log10(MW_5_EQS_PER_YEAR/10**(np.min(CSEP_MW_BINS)-5)/aftershock_region.dh/aftershock_region.dh)
    vmin =np.log10(MW_5_EQS_PER_YEAR/10**(np.max(CSEP_MW_BINS)-5)/aftershock_region.dh/aftershock_region.dh)
    ax = plot_spatial_dataset(log_cond_rate_dens_cart,
                              aftershock_region,
                              plot_args={'clabel': r'Log$_{10}$ Conditional Rate Density',
                                         'clim': [vmin, vmax],
                                         'title': f'Approximate Rate Density with Observations\nMw > {mw_min}'})
    ax.scatter(comcat.get_longitudes(), comcat.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
    crd_fname = build_figure_filename(plot_dir, mw_min, 'crd_obs')
    ax.figure.savefig(crd_fname)
    t1 = time.time()
    print(f'Done with Spatial tests in {t1-t0} seconds.')

    # compute interevent-time distribution test
    t0 = time.time()
    print('Computing Inter-Event Time Test')
    ietd_result = interevent_time_test(u3_filt, comcat)
    ietd_test_fname = build_figure_filename(plot_dir, mw_min, 'ietd_test')
    _ = plot_distribution_test(ietd_result, show=False, plot_args={'percentile': 95,
                                                                    'title': f'Inter-event Time Distribution-Test\nMw>{mw_min}',
                                                                    'bins': 'auto',
                                                                    'xlabel': "D* Statistic",
                                                                    'ylabel': r"P(X $\leq$ x)",
                                                                    'filename': ietd_test_fname})
    t1 = time.time()
    print(f"Finished IETD Test in {t1-t0} seconds")

    # compute interevent-distance distribution test
    t0 = time.time()
    print('Computing Inter-Event Distance Test')
    iedd_result = interevent_distance_test(u3_filt, comcat)
    iedd_test_fname = build_figure_filename(plot_dir, mw_min, 'iedd_test')
    _ = plot_distribution_test(iedd_result, show=False, plot_args={'percentile': 95,
                                                                    'title': f'Inter-event Distance Distribution-Test\nMw>{mw_min}',
                                                                    'bins': 'auto',
                                                                    'xlabel': "D* Statistic",
                                                                    'ylabel': r"P(X $\leq$ x)",
                                                                    'filename': iedd_test_fname})
    t1 = time.time()
    print(f"Finished IETD Test in {t1-t0} seconds")

    print('Computing Total Event Rate Distribution Test')
    t0 = time.time()
    terd_result = interevent_distance_test(u3_filt, comcat)
    terd_test_fname = build_figure_filename(plot_dir, mw_min, 'terd_test')
    _ = plot_distribution_test(terd_result, show=False, plot_args={'percentile': 95,
                                                                    'title': f'Total Event Rate Distribution-Test\nMw>{mw_min}',
                                                                    'bins': 'auto',
                                                                    'xlabel': "D* Statistic",
                                                                    'ylabel': r"P(X $\leq$ x)",
                                                                    'filename': terd_test_fname})
    t1 = time.time()
    print(f"Finished IETD Test in {t1-t0} seconds")

    # compute b-value test
    print('Computing B-Value Test')
    t0 = time.time()
    bv_result = bvalue_test(u3_filt, comcat)
    bv_test_fname = build_figure_filename(plot_dir, mw_min, 'bv_test')
    _ = plot_number_test(bv_result, show=False, plot_args={'percentile': 95,
                                                           'title': f"B-Value Distribution Test\nMw>{mw_min}",
                                                           'bins': 'auto',
                                                           'filename': bv_test_fname})
    t1 = time.time()
    print(f"Finished IETD Test in {t1-t0} seconds")
    # compile results
    results['forecast_name'] = u3_filt[0].name
    results['n-test'] = n_test_result
    results['m-test'] = m_test_result
    results['l-test'] = l_test_result
    results['s-test'] = s_test_result
    results['ietd-test'] = ietd_result
    results['iedd-test'] = iedd_result
    results['terd-test'] = terd_result
    results['bv-test'] = bv_result
    results['terd-test_plot'] = get_relative_path(terd_test_fname)
    results['iedd-test_plot'] = get_relative_path(iedd_test_fname)
    results['ietd-test_plot'] = get_relative_path(ietd_test_fname)
    results['bv-test_plot'] = get_relative_path(bv_test_fname)
    results['n-test_plot'] = get_relative_path(n_test_fname)
    results['cum_plot'] = get_relative_path(cum_counts_fname)
    results['m-test_plot'] = get_relative_path(m_test_fname)
    results['mag_hist'] = get_relative_path(mag_hist_fname)
    results['l-test_plot'] = get_relative_path(l_test_fname)
    results['s-test_plot'] = get_relative_path(s_test_fname)
    results['crd_obs'] = get_relative_path(crd_fname)
    return results

def build_figure_filename(dir, mw, plot_id):
    basename = f"{plot_id}_{mw}.png"
    return os.path.join(dir, basename)

def generate_table_from_results(results, tests=('n-test', 'm-test', 's-test', 'l-test', 'ietd-test', 'bv-test', 'iedd-test')):
    table = []
    header = list(results['minimum_mw'])
    header.insert(0, ' ')
    table.append(tuple(header))
    for test in tests:
        row = []
        row.append('<b>' + test + '</b>')
        test_results = results[test]
        for result in test_results:
            row.append(result.quantile)
        table.append(tuple(row))
    return table

if __name__ == "__main__":
    # mainshock information
    import matplotlib
    import json

    t0 = time.time()
    # need non-interactive backend for multiprocessing stuff
    matplotlib.use('agg')
    event_id = 'ci38457511' # mw 7.1
    # event_id = 'ci38443183' # mw 6.4
    event = get_event_by_id(event_id)
    # starting download 0.5 seconds after mainshock
    event_lon, event_lat = event.longitude, event.latitude
    event_mw = event.magnitude
    # simulation_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m64_point_src'
    simulation_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt'
    plot_dir = os.path.join(simulation_dir, 'plotting')
    # this defo needs to be a class
    with open(os.path.join(simulation_dir, 'config.json')) as f:
        u3etas_config = json.load(f)
    origin_time = u3etas_config['startTimeMillis']
    # some information for the main program not necessarily just the u3etas config, maybe event isn't in there
    config = {
              'numSimulations': 100000,
              'startTimeMillis': origin_time,
              'magnitude': event_mw,
              'eventLongitude': event_lon,
              'eventLatitude': event_lat,
              'eventMagnitude': event_mw,
              'eventId': event_id,
              'plottingDirectory': plot_dir,
              'simulationName': u3etas_config['simulationName']}
    # run evaluation until present time
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    # evaluate in small mag bins, and then total
    mw_min = [3.0, 3.5, 4.0]
    # freeze the configuration and the end_time
    partial_main = functools.partial(main, sim_dir=simulation_dir, config=config, end_time=now)
    # parallelize over the magnitudes
    results = []
    # this is actually so dumb
    for mw in mw_min:
        output = partial_main(mw)
        results.append(output)

    # combine results into single dict, where each key holds a tuple
    combined_results = {
        k: tuple(d.get(k) for d in results)
        for k in set().union(*results)
    }
    # create the notebook for results
    notebook = ResultsNotebook()
    # introduction is fixed,  might change to make more general
    notebook.add_introduction(adict={'simulation_name': u3etas_config['simulationName'],
                                    'origin_time': epoch_time_to_utc_datetime(origin_time),
                                    'evaluation_time': now,
                                    'catalog_source': 'Comcat',
                                    'forecast_name': combined_results['forecast_name'][0],
                                    'num_simulations': config['numSimulations']})
    notebook.add_sub_heading('Visual Overview of Forecast', 1, "")
    # add cumulative event counts plot, notice the tuple is wrapped in a list, indicates 1 row
    notebook.add_result_figure('Cumulative Event Counts', 2, [combined_results['cum_plot']])
    # Magnitude histogram (only plot the one for minimum magnitude, results are ordered by map)
    notebook.add_result_figure('Magnitude Histogram', 2, combined_results['mag_hist'][0])
    notebook.add_result_figure('Conditional Rate Density with Observations', 2, [combined_results['crd_obs']])
    notebook.add_sub_heading('CSEP Consistency Tests', 1, "<b>Note</b>: These tests are still in development. Feedback appreciated.")
    notebook.add_result_figure('Number Test', 2, [combined_results['n-test_plot']])
    notebook.add_result_figure('Magnitude Test', 2, [combined_results['m-test_plot']])
    notebook.add_result_figure('Spatial Test', 2, [combined_results['s-test_plot']])
    notebook.add_result_figure('Likelihood Test', 2, [combined_results['l-test_plot']])
    notebook.add_sub_heading('One-point Statistics', 1, "")
    notebook.add_result_figure('B-Value Test', 2, [combined_results['bv-test_plot']])
    notebook.add_sub_heading('Distribution Statistics', 1, "")
    notebook.add_result_figure('Inter-event Distance Distribution', 2, [combined_results['iedd-test_plot']])
    notebook.add_result_figure('Inter-event Time Distribution', 2, [combined_results['ietd-test_plot']])
    notebook.add_result_figure('Total Event Rate Distribution', 2, [combined_results['terd-test_plot']])
    # table should be a list of tuples, where each tuple is the row
    table = generate_table_from_results(combined_results)
    # assemble the results from the dictionary
    notebook.add_sub_heading('Results Summary', 1, notebook.get_table(table))
    # calling this produces the toc, and writes output.
    notebook.finalize(simulation_dir)

    t1 = time.time()
    print(f'Completed all processing in {t1-t0} seconds.')