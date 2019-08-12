import copy
import datetime
import os
import sys
from collections import defaultdict
from itertools import tee

import tqdm

import time
from csep import load_stochastic_event_sets, load_comcat
from csep.utils.time import utc_now_epoch, datetime_to_utc_epoch, epoch_time_to_utc_datetime
from csep.utils.spatial import masked_region, california_relm_region
from csep.utils.basic_types import Polygon
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.comcat import get_event_by_id
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR
from csep.core.evaluations import NumberTest, MagnitudeTest, LikelihoodAndSpatialTest, CumulativeEventPlot, \
    MagnitudeHistogram, ConditionalRatePlot, BValueTest, TotalEventRateDistribution, \
    InterEventDistanceDistribution, InterEventTimeDistribution, SpatialLikelihoodPlot

# MAIN SCRIPT BELOW HERE

# set up basic stuff, most of this stuff coming from ucerf3
# event_id = 'ci38443183' # mw 6.4 ridgecrest, should be in later editions of ucerf3 json format
# filename = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m64_point_src/results_complete.bin'
# plot_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m64_point_src/plotting'
event_id = 'ci38457511' # mw 7.1
filename = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt/results_complete.bin'
plot_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt/plotting'
n_cat = 1000
event = get_event_by_id(event_id)
origin_epoch = datetime_to_utc_epoch(event.time)
rupture_length = WellsAndCoppersmith.mag_length_strike_slip(event.magnitude) * 1000
aftershock_polygon = Polygon.from_great_circle_radius((event.longitude, event.latitude),
                                                      3*rupture_length, num_points=100)
aftershock_region = masked_region(california_relm_region(), aftershock_polygon)
end_epoch = utc_now_epoch()
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

# these classes must implement process_catalog() and evaluate()
# calling evaluate should return an EvaluationResult namedtuple
data_products = {
     # needs event count per catalog
     'n-test': NumberTest(),
     'm-test': MagnitudeTest(),
     'l-test': LikelihoodAndSpatialTest(),
     'cum-plot': CumulativeEventPlot(origin_epoch, end_epoch),
     'mag-hist': MagnitudeHistogram(calc=False),
     'crd-plot': ConditionalRatePlot(calc=False),
     'bv-test': BValueTest(),
     'like-plot': SpatialLikelihoodPlot(calc=False)
     # 'terd-test': TotalEventRateDistribution(),
     # 'iedd-test': InterEventDistanceDistribution(),
     # 'ietd-test': InterEventTimeDistribution()
}

t0 = time.time()
u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name='UCERF3-ETAS', region=aftershock_region)
for i, cat in tqdm.tqdm(enumerate(u3), total=n_cat, position=0):
    cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region)
    for name, calc in data_products.items():
        calc.process_catalog(copy.copy(cat_filt))
    if (i+1) % n_cat == 0:
        break
t1 = time.time()
print(f'Processed catalogs in {t1-t0} seconds')

# share data where applicable
data_products['mag-hist'].data = data_products['m-test'].data
data_products['crd-plot'].data = data_products['l-test'].data
data_products['like-plot'].data = data_products['l-test'].data

# finalizes
results = defaultdict(list)
for name, calc in data_products.items():
    print(f'Finalizing calculations for {name} and plotting')
    result = calc.evaluate(comcat, args=(u3, time_horizon, end_epoch, n_cat))
    # store results for later, maybe?
    results[name].append(result)
    # plot, and store in plot_dir
    calc.plot(result, plot_dir, show=False)
t2 = time.time()
print(f"Evaluated forecasts in {t2-t1} seconds")
print(f"Finished everything in {t2-t0} seconds with average time per catalog of {(t2-t0)/n_cat} seconds")

# build report with custom layout