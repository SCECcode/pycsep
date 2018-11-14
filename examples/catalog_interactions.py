import time
import os
import numpy
import matplotlib.pyplot as pyplot

from csep.core.catalogs import UCERF3Catalog, ComcatCatalog
from csep.utils.plotting import plot_cumulative_events_versus_time, plot_magnitude_versus_time

"""
Note:
    This script requires about 12-14Gb of Ram because generators are not implemented for the plots.
"""

# UCERF3 Synthetics
ucerf3_numbers = []
nofaults_numbers = []
min_magnitude = []

project_root = '/Users/wsavran/Projects/CSEP2/u3etas_simulations/landers_experiment'
filename = os.path.join(project_root, '10-23-2018_landers-pt1/results_complete.bin')
filename_nofaults = os.path.join(project_root, '10-31-2018_landers-nofaults-pt1/results_complete.bin')

t0 = time.time()
u3catalogs = list(UCERF3Catalog.load_catalogs(filename=filename, name='UCERF3-ETAS'))
for u3catalog in u3catalogs:
    if u3catalog.catalog_id % 500 == 0:
        print('Loaded {} catalogs'.format(u3catalog.catalog_id))
    ucerf3_numbers.append(u3catalog.get_number_of_events())
t1 = time.time()
print('Loaded {} UCERF3 catalogs in {} seconds.\n'.format(u3catalog.catalog_id+1, (t1-t0)))

t0 = time.time()
u3catalogs_nf = list(UCERF3Catalog.load_catalogs(filename=filename_nofaults, name='UCERF3-NoFaultsETAS'))
for u3catalog_nf in u3catalogs_nf:
    if u3catalog_nf.catalog_id % 500 == 0:
        print('Loaded {} catalogs'.format(u3catalog_nf.catalog_id))
    nofaults_numbers.append(u3catalog_nf.get_number_of_events())
t1 = time.time()
print('Loaded {} UCERF3 catalogs in {} seconds.\n'.format(u3catalog_nf.catalog_id+1, (t1-t0)))
#
# Comcat Synthetics
epoch_time = 709732655000
duration_in_years = 1.0
t0 = time.time()
comcat = ComcatCatalog(start_epoch=epoch_time, duration_in_years=1.0, name='Comcat',
                            min_magnitude=2.55,
                            min_latitude=31.50, max_latitude=43.00,
                            min_longitude=-125.40, max_longitude=-113.10,)
t1 = time.time()
print("Fetched Comcat catalog in {} seconds.\n".format(t1-t0))
print("Downloaded Comcat Catalog with following parameters")
print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
print("Min Magnitude: {}\n".format(comcat.min_magnitude))

comcat_count = comcat.get_number_of_events()

# Statements about Catalog Statistics
print("Found {} events in the Comcat catalog.".format(comcat_count))

print("Found {} events in the UCERF3 catalog with lowest number of events.".format(numpy.min(ucerf3_numbers)))
print("Found {} events in the UCERF3 catalog with max number of events.".format(numpy.max(ucerf3_numbers)))
print("In UCERF3 the median events were {} and the mean events were {}."
      .format(numpy.median(ucerf3_numbers),numpy.mean(ucerf3_numbers)))

print("Found {} events in the UCERF3-NoFaults catalog with lowest number of events.".format(numpy.min(nofaults_numbers)))
print("Found {} events in the UCERF3 catalog with max number of events.".format(numpy.max(nofaults_numbers)))
print("In UCERF3-Nofaults the median events were {} and the mean events were {}."
      .format(numpy.median(nofaults_numbers),numpy.mean(nofaults_numbers)))

# Plotting
ucerf3_numbers = numpy.array(ucerf3_numbers)
nofaults_numbers = numpy.array(nofaults_numbers)
fig = pyplot.figure()
pyplot.hist(ucerf3_numbers, bins=60, color='blue', edgecolor='black', alpha=0.7, label='UCERF3-ETAS')
pyplot.hist(nofaults_numbers, bins=60, color='green', edgecolor='black', alpha=0.7, label='UCERF3-NoFaults')
pyplot.axvline(x=comcat_count, linestyle='--', color='black', label='Comcat')
pyplot.xlabel('Event Count')
pyplot.ylabel('Frequency')
pyplot.xlim([0, numpy.max(numpy.vstack((ucerf3_numbers, nofaults_numbers)))])
pyplot.legend(loc='best')
# fig.savefig(os.path.join(project_root, 'u3nofaults_p_withfaults_hist-ntest_mw4.0.pdf'))
pyplot.show()

# Plot cumulative events
fig, ax = plot_cumulative_events_versus_time(u3catalogs, comcat)
plot_cumulative_events_versus_time(u3catalogs_nf, comcat, fig=fig)

# Plot magnitude versus time
plot_magnitude_versus_time(comcat)
plot_magnitude_versus_time(u3catalog, show=True)
