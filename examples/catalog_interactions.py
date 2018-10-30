import time
import numpy
import matplotlib.pyplot as pyplot

from csep.core.catalogs import UCERF3Catalog, ComcatCatalog

# UCERF3 Synthetics
ucerf3_numbers = []
min_magnitude = []
t0 = time.time()
filename='/Users/wsavran/Projects/CSEP2/u3etas_simulations/landers_experiment/10-23-2018_landers-pt1/results_complete.bin'
for u3catalog in UCERF3Catalog.load_catalogs(filename=filename):
    if u3catalog.catalog_id % 500 == 0:
        print('Loading catalog {}.'.format(u3catalog.catalog_id))
    ucerf3_numbers.append(u3catalog.get_number_of_events())

    # print minimum magnitude
    min_magnitude.append(numpy.min(u3catalog.catalog['magnitude']))

t1 = time.time()
print('Loaded {} UCERF3 catalogs in {} seconds.\n'.format(u3catalog.catalog_id+1, (t1-t0)))

# Comcat Synthetics
epoch_time = 709732655000
duration_in_years = 1.0
comcat = ComcatCatalog(start_epoch=epoch_time, duration_in_years=1.0, lazy_load=False)
print("Downloaded Comcat Catalog with following parameters:")
print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
print("Min Magnitude: {}\n".format(comcat.min_magnitude))
comcat_count = comcat.get_number_of_events()
print("Found {} events in the Comcat catalog.".format(comcat_count))
print("Found {} events in the UCERF3 catalog with lowest number of events.".format(numpy.min(ucerf3_numbers)))

# Plotting
pyplot.figure()
pyplot.hist(ucerf3_numbers, bins=60, color='blue', edgecolor='black', alpha=0.7)
pyplot.axvline(x=comcat_count, linestyle='--', color='black')
pyplot.xlabel('Event Count')
pyplot.ylabel('Frequency')
pyplot.show()




