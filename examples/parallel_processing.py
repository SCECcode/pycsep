import datetime

from dask import compute

import time

from dask.delayed import delayed
from dask.distributed import Client

from csep.utils.comcat import get_event_by_id
from csep.utils.time import datetime_to_utc_epoch, utc_now_epoch
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.basic_types import Polygon
from csep.utils.spatial import masked_region, california_relm_region
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR

from csep.utils.parallel import do_evaluate_catalogs

from csep.core.evaluations import NumberTest, MagnitudeTest, LikelihoodAndSpatialTest, CumulativeEventPlot, \
    MagnitudeHistogram, ApproximateRatePlot, BValueTest, TotalEventRateDistribution, InterEventDistanceDistribution, \
    InterEventTimeDistribution

# some basic setup that could easily be looped into the function itself, we will see about that.
# potentially start up in the Processing class.
event_id = 'ci38457511' # mw 7.1
filename = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt/results_complete.bin'
plot_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_finite_flt/plotting'
n_cat = 10000
event = get_event_by_id(event_id)
origin_epoch = datetime_to_utc_epoch(event.time)
rupture_length = WellsAndCoppersmith.mag_length_strike_slip(event.magnitude) * 1000
aftershock_polygon = Polygon.from_great_circle_radius((event.longitude, event.latitude),
                                                      3*rupture_length, num_points=100)
aftershock_region = masked_region(california_relm_region(), aftershock_polygon)
end_epoch = utc_now_epoch()
time_horizon = (end_epoch - origin_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000
event_time = event.time.replace(tzinfo=datetime.timezone.utc)

task_configuration = {
    'region': aftershock_region,
    'sim_type': 'ucerf3',
    'name': 'UCERF3-ETAS',
    'start_epoch': origin_epoch,
    'end_epoch': end_epoch,
    'n_cat': n_cat,
    'plot_dir': plot_dir
}

data_products = {
     # needs event count per catalog
     'n-test': NumberTest(),
     # magnitude distribution
     'm-test': MagnitudeTest(),
     # gridded counts or evnets
     'l-test': LikelihoodAndSpatialTest(),
     # time-binned event counts
     'cum-plot': CumulativeEventPlot(origin_epoch, end_epoch),
     # # magnitude distribution (same as m-test)
     'mag-hist': MagnitudeHistogram(),
     # # (same as l-test
     'crd-plot': ApproximateRatePlot(),
     # # b-value per catalog
     'bv-test': BValueTest()
     # # same as l-test
     # 'terd-test': TotalEventRateDistribution(),
     # # inter-event distances
     # 'iedd-test': InterEventDistanceDistribution(),
     # # inter-event times
     # 'ietd-test': InterEventTimeDistribution()
}

# try launching using dask delayed
client = Client()
results = []
for k, task in data_products.items():
    out = delayed(do_evaluate_catalogs)(filename, task, task_configuration)
    results.append(out)
results = compute(*results)