# native imports
import time
import datetime
import os

# 3rd party imports
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# CSEP Imports
import csep
from csep.core.evaluations import number_test, magnitude_test
from plotting import plot_number_test, plot_magnitude_test, plot_likelihood_test, plot_spatial_test
from csep.core.evaluations import combined_likelihood_and_spatial
from csep.utils.plotting import plot_magnitude_histogram
from csep.utils.time import datetime_to_utc_epoch
from csep.utils.constants import CSEP_MW_BINS, SECONDS_PER_ASTRONOMICAL_YEAR, MW_5_EQS_PER_YEAR
from csep.utils.spatial import california_relm_region
from csep.utils.cmath import discretize
from csep.utils.plotting import plot_spatial_dataset

def main(config, mw_min, end_time):

    # set things up
    filename = os.path.join(config['outputDir'], 'results_complete.bin')
    relm_region = california_relm_region()
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    now_epoch = csep.utils.time.datetime_to_utc_epoch(now)
    # time-horizon of forecast, epochs are in milliseconds
    time_horizon = (now_epoch - origin_time) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000
    n_cat = config['numSimulations']
    # Download comcat catalog
    print('Loading Comcat.')
    t0 = time.time()
    comcat = csep.core.catalogs.ComcatCatalog(start_epoch=origin_time, end_time=end_time,
                                              name='Comcat', min_magnitude=2.55,
                                              min_latitude=31.50, max_latitude=43.00,
                                              min_longitude=-125.40, max_longitude=-113.10, region=relm_region, query=True)
    comcat = comcat.filter(f'magnitude >= {mw_min}')
    t1 = time.time()
    print("Fetched Comcat catalog in {} seconds.\n".format(t1 - t0))
    print("Downloaded Comcat Catalog with following parameters")
    print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
    print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
    print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
    print("Min Magnitude: {}\n".format(comcat.min_magnitude))
    print(f"Found {comcat.get_number_of_events()} events in the Comcat catalog.")
    print(f'Proceesing Catalogs.')

    # Process Stochastic Event Sets
    gridded_counts = np.zeros(relm_region.num_nodes)
    mags = np.zeros((n_cat, len(CSEP_MW_BINS)))
    counts = np.zeros(n_cat)
    u3_filt = []

    u3catalogs = csep.load_stochastic_event_set(filename=filename, type='ucerf3', format='native', name='UCERF3-ETAS', region=relm_region)
    # read catalogs from disk
    for i in tqdm.tqdm(range(n_cat), total=n_cat):
        cat = next(u3catalogs)
        cat_filt = cat.filter(f'origin_time < {now_epoch}').filter(f'magnitude >= {mw_min}')
        # store catalog will need to iterate over these again to compute pseudo-likelihood
        u3_filt.append(cat_filt)
        # counts on spatial grid
        gridded_counts += cat_filt.gridded_event_counts()
        # counts in mag bins
        mags[i,:] += cat_filt.binned_magnitude_counts()
        # global counts
        counts[i] = cat_filt.event_count

    print("Computing N-test results.")
    t0 = time.time()
    n_test_result = number_test(u3_filt, comcat)
    _ = plot_number_test(n_test_result, show=False, plot_args={'percentile': 95,
                                                                'title': f'N-Test Following M7.1 until Present\nMw>{mw_min}',
                                                                'bins': 'auto'})
    t1 = time.time()
    print(f'Done with N-test in {t1-t0} seconds.')

    print("Computing M-test results.")
    t0 = time.time()
    m_test_result = magnitude_test(u3_filt, comcat)
    _ = plot_magnitude_test(m_test_result, show=False, plot_args={'percentile': 95,
                                                                   'title': f'M-Test Following M7.1 until Present\nMw>{mw_min}',
                                                                   'bins': 'auto'})
    print("Plotting Magnitude Histogram")
    _ = plot_magnitude_histogram(u3_filt, comcat, show=False, plot_args={'xlim': [mw_min, np.max(CSEP_MW_BINS)],
                                                                         'sim_label': u3_filt[0].name})
    t1 = time.time()
    print(f'Done with M-test in {t1-t0} seconds.')

    print("Computing S-test and L-test results.")
    t0 = time.time()
    cond_rate_dens = gridded_counts/n_cat/relm_region.dh/relm_region.dh/time_horizon
    l_test_result, s_test_result = combined_likelihood_and_spatial(u3_filt, comcat, cond_rate_dens, time_horizon)
    print(f"{l_test_result.name} result is {l_test_result.status}")
    _ = plot_likelihood_test(l_test_result, show=False,
                              plot_args={'percentile': 95,
                                         'title': f'L-Test Following M7.1 until Present\nMw>{mw_min}',
                                         'bins': 'auto'})
    t1 = time.time()
    print(f'Done with L-test in {t1-t0} seconds.')

    print("Computing S-test results.")
    t0 = time.time()
    print(f"{s_test_result.name} result is {s_test_result.status}")
    _ = plot_spatial_test(s_test_result, show=False,
                           plot_args={'percentile': 95,
                                      'title': f'S-Test Following M7.1 until Present\nMw>{mw_min}',
                                      'bins': 'auto'})
    print("Plotting spatial distribution of events.")
    # used for plotting, converted to 2d grid
    log_cond_rate_dens_cart = relm_region.get_cartesian(np.log10(cond_rate_dens))
    log_cond_rate_dens_cart[log_cond_rate_dens_cart == -np.inf] = -9999
    # compute color bar given rate of certain magnitude events, hard-coded for CSEP MW Bins
    vmax = np.log10(MW_5_EQS_PER_YEAR/10**(np.min(CSEP_MW_BINS)-5)/relm_region.dh/relm_region.dh)
    vmin =np.log10(MW_5_EQS_PER_YEAR/10**(np.max(CSEP_MW_BINS)-5)/relm_region.dh/relm_region.dh)
    ax = plot_spatial_dataset(log_cond_rate_dens_cart, relm_region,
                              plot_args={'clabel': r'Log$_{10}$ Conditional Rate Density',
                                         'clim': [vmin, vmax]})
    ax.scatter(comcat.get_longitudes(), comcat.get_latitudes(),
               marker='.', color='white', s=40, edgecolors='black')
    t1 = time.time()
    print(f'Done with Spatial tests in {t1-t0} seconds.')
    # show all plots
    plt.show()

if __name__ == "__main__":
    # these items will come from the forecast configuration
    origin_time = 1562383194000
    config = {'outputDir': '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt',
              "numSimulations": 1000,
              'startTimeMillis': origin_time}
    # run evaluation until present time
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    # minimum magnitude
    mw_min = 3.4999
    main(config, mw_min, now)