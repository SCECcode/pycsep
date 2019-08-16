import copy
import datetime
import matplotlib
import os
import json

import tqdm
import seaborn as sns

import time
from csep import load_stochastic_event_sets, load_comcat
from csep.utils.time import utc_now_epoch, datetime_to_utc_epoch, epoch_time_to_utc_datetime
from csep.utils.spatial import masked_region, california_relm_region
from csep.utils.basic_types import Polygon
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.comcat import get_event_by_id
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR
from csep.core.evaluations import NumberTest, MagnitudeTest, LikelihoodAndSpatialTest, CumulativeEventPlot, \
    MagnitudeHistogram, ApproximateRatePlot, BValueTest, SpatialLikelihoodPlot, ConditionalApproximateRatePlot
from csep.utils.file import get_relative_path, mkdirs
from csep.utils.documents import MarkdownReport

matplotlib.use('agg')
matplotlib.rcParams['figure.max_open_warning'] = 150
sns.set()

def ucerf3_consistency_testing(sim_dir, event_id, end_epoch, n_cat=None, plot_dir=None):
    """
    computes all csep consistency tests for simulation located in sim_dir with event_id

    Args:
        sim_dir (str): directory where results and configuration are stored
        event_id (str): event_id corresponding to comcat event
        data_products (dict):

    Returns:

    """
    # set up directories
    filename = os.path.join(sim_dir, 'results_complete.bin')
    if plot_dir is None:
        plot_dir = os.path.join(sim_dir, 'plots')
    config_file = os.path.join(sim_dir, 'config.json')
    mkdirs(plot_dir)

    # load ucerf3 configuration
    with open(os.path.join(config_file), 'r') as f:
        u3etas_config = json.load(f)

    # determine how many catalogs to process
    if n_cat is None:
        n_cat = u3etas_config['numSimulations']

    # download comcat information
    event = get_event_by_id(event_id)

    # filter to aftershock radius
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(event.magnitude) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((event.longitude, event.latitude),
                                                          3*rupture_length, num_points=100)
    aftershock_region = masked_region(california_relm_region(), aftershock_polygon)

    # event timing
    origin_epoch = datetime_to_utc_epoch(event.time)
    time_horizon = (end_epoch - origin_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000
    event_time = event.time.replace(tzinfo=datetime.timezone.utc)

    # Download comcat catalog
    print('Loading Comcat.')
    comcat = load_comcat(event_time, epoch_time_to_utc_datetime(end_epoch),
                              min_magnitude=2.50,
                              min_latitude=31.50, max_latitude=43.00,
                              min_longitude=-125.40, max_longitude=-113.10)
    comcat = comcat.filter_spatial(aftershock_region)
    print(comcat)

    # define products to compute on simulation
    data_products = {
         'n-test': NumberTest(),
         'm-test': MagnitudeTest(),
         'l-test': LikelihoodAndSpatialTest(),
         'cum-plot': CumulativeEventPlot(origin_epoch, end_epoch),
         'mag-hist': MagnitudeHistogram(calc=False),
         'arp-plot': ApproximateRatePlot(calc=False),
         'bv-test': BValueTest(),
         'like-plot': SpatialLikelihoodPlot(calc=False),
         'carp-plot': ConditionalApproximateRatePlot(comcat),
    }

    print(f'Will process {n_cat} catalogs from simulation\n')
    for k, v in data_products.items():
        print(f'Computing {v.__class__.__name__}')

    # read the catalogs
    print('Begin processing catalogs')
    t0 = time.time()
    u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name='UCERF3-ETAS', region=aftershock_region)
    for i, cat in enumerate(u3):
        cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region)
        for name, calc in data_products.items():
            calc.process_catalog(copy.copy(cat_filt))
        if (i+1) % n_cat == 0:
            break

        if (i+1) % 2500 == 0:
            t1 = time.time()
            print(f'Processed {i+1} catalogs in {t1-t0} seconds')
    t2 = time.time()
    print(f'Finished processing catalogs in {t2-t0} seconds')

    # share data where applicable
    data_products['mag-hist'].data = data_products['m-test'].data
    data_products['arp-plot'].data = data_products['l-test'].data
    data_products['like-plot'].data = data_products['l-test'].data

    # evaluate the catalogs and store results
    t1 = time.time()
    results = {}
    for name, calc in data_products.items():
        print(f'Finalizing calculations for {name} and plotting')
        result = calc.evaluate(comcat, args=(u3, time_horizon, end_epoch, n_cat))
        # store results for later, maybe?
        results[name] = result
        # plot, and store in plot_dir
        calc.plot(result, plot_dir, show=False)
    t2 = time.time()
    print(f"Evaluated forecasts in {t2-t1} seconds")
    print(f"Finished everything in {t2-t0} seconds with average time per catalog of {(t2-t0)/n_cat} seconds")

    # create the notebook for results
    md = MarkdownReport('results_benchmark.ipynb')

    md.add_introduction(adict={'simulation_name': u3etas_config['simulationName'],
                                    'origin_time': epoch_time_to_utc_datetime(origin_epoch),
                                    'evaluation_time': epoch_time_to_utc_datetime(end_epoch),
                                    'catalog_source': 'Comcat',
                                    'forecast_name': 'UCERF3-ETAS',
                                    'num_simulations': n_cat})

    md.add_sub_heading('Visual Overview of Forecast', 1, "")
    md.add_result_figure('Cumulative Event Counts', 2, list(map(get_relative_path, data_products['cum-plot'].fnames)), ncols=2)
    md.add_result_figure('Magnitude Histogram', 2, list(map(get_relative_path, data_products['mag-hist'].fnames)))
    md.add_result_figure('Approximate Rate Density with Observations', 2, list(map(get_relative_path, data_products['arp-plot'].fnames)), ncols=2)
    md.add_result_figure('Normalized Likelihood Per Event', 2, list(map(get_relative_path, data_products['like-plot'].fnames)), ncols=2)
    md.add_result_figure('Conditional Rate Density', 2, list(map(get_relative_path, data_products['carp-plot'].fnames)), ncols=2)

    md.add_sub_heading('CSEP Consistency Tests', 1, "<b>Note</b>: These tests are still in development. Feedback appreciated.")
    md.add_result_figure('Number Test', 2, list(map(get_relative_path, data_products['n-test'].fnames)))
    md.add_result_figure('Magnitude Test', 2, list(map(get_relative_path, data_products['m-test'].fnames)))
    md.add_result_figure('Spatial Test', 2, list(map(get_relative_path, data_products['l-test'].fnames['s-test'])))
    md.add_result_figure('Likelihood Test', 2, list(map(get_relative_path, data_products['l-test'].fnames['l-test'])))

    md.add_sub_heading('One-point Statistics', 1, "")
    md.add_result_figure('B-Value Test', 2, list(map(get_relative_path, data_products['bv-test'].fnames)))
    md.finalize(sim_dir)

    t1 = time.time()
    print(f'Completed all processing in {t1-t0} seconds')

if __name__ == "__main__":
    sim_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_shakemap_src'
    event_id = 'ci38457511'
    end_epoch = utc_now_epoch()
    ucerf3_consistency_testing(sim_dir, event_id, end_epoch)