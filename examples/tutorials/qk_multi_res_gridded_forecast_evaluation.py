"""

.. _quadtree_multi-res_grid-forecast-evaluation:

Quadtree Multi-resolution Grid-based Forecast Evaluation
=======================================

This example demonstrates how to create a multi-resolution grid based on earthquake catalog. Then use that grid to create and evaluate a time-independent forecast. Grid-based
forecasts assume the variability of the forecasts is Poissonian. Therefore, Poisson-based evaluations
should be used to evaluate grid-based forecasts.

Overview:
    1. Define spatial grid
    2. Load forecast
    3. Load evaluation catalog
    4. Apply Poissonian evaluations for grid-based forecasts
    5. Visualize evaluation results
"""

####################################################################################################################################
# Load required libraries
# -----------------------
#
# Most of the core functionality can be imported from the top-level :mod:`csep` package. Utilities are available from the
# :mod:`csep.utils` subpackage.
import numpy
import pandas
from csep.core import poisson_evaluations as poisson
from csep.utils import time_utils, plots
from csep.core.regions import QuadtreeGrid2D
from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year_to_utc_epoch
from csep.core.catalogs import CSEPCatalog

####################################################################################################################################
# Load Training Catalog
# --------------------------
#
# We define a multi-resolution quadtree using earthquake catalog. We load a training catalog in CSEP and use that catalog to create a multi-resolution grid.
# Sometimes, we do not the catalog in exact format as requried by PyCSEP. So we can read a catalog using Pandas and convert it 
# into the format accepable by PyCSEP. Then we instantiate an object of class CSEPCatalog by calling function :func:`csep.core.regions.CSEPCatalog.from_dataframe`

dfcat = pandas.read_csv('cat_train_2013.csv')
column_name_mapper = {
    'lon': 'longitude',
    'lat': 'latitude',
    'mag': 'magnitude',
    'index': 'id'
    }

# maps the column names to the dtype expected by the catalog class
dfcat = dfcat.reset_index().rename(columns=column_name_mapper)
# create the origin_times from decimal years
dfcat['origin_time'] = dfcat.apply(lambda row: decimal_year_to_utc_epoch(row.year), axis=1)

# create catalog from dataframe
catalog_train = CSEPCatalog.from_dataframe(dfcat)
print(catalog_train)

####################################################################################################################################
# Define Multi-resolution Gridded Region
# ------------------------------------------------
# Now use define a threshold for maximum number of earthquake allowed per cell, i.e. Nmax
# and call :func:`csep.core.regions.QuadtreeGrid_from_catalog` to create a multi-resolution grid.
# For simplicity we assume only single magnitude bin, i.e. all the earthquakes greater than and equal to 5.95

mbins = numpy.array([5.95])
Nmax = 10
r = QuadtreeGrid2D.from_catalog(catalog_train, Nmax, magnitudes=mbins)
print('Number of cells in grid :', r.num_nodes)

####################################################################################################################################
# Load forecast
# -------------
# An example time-independent forecast had been created for this grid and provided the example forecast data set along with the main repository.
# We load the time-independent global forecast which has time horizon of 1 year.   
# The filepath is relative to the root directory of the package. You can specify any file location for your forecasts.

forecast_data = numpy.loadtxt('example_rate_zoom=EQ10L11.csv')
#Reshape forecast as Nx1 array
forecast_data = forecast_data.reshape(-1,1)

forecast_gridded = GriddedForecast(data = forecast_data, region = r, magnitudes = mbins, name = 'Example Multi-res Forecast')

#The loaded forecast is for 1 year. The test catalog we will use to evaluate is for 6 years. So we can rescale the forecast.
print(f"expected event count before scaling: {forecast_gridded.event_count}")
forecast_gridded.scale(6)
print(f"expected event count after scaling: {forecast_gridded.event_count}")
    

####################################################################################################################################
# Load evaluation catalog
# -----------------------
#
# We have a test catalog stored here. We can read the test catalog as a pandas frame and convert it into a format that is acceptable to PyCSEP
# Then we instantiate an object of catalog

dfcat = pandas.read_csv('cat_test.csv')

column_name_mapper = {
    'lon': 'longitude',
    'lat': 'latitude',
    'mag': 'magnitude'
    }

# maps the column names to the dtype expected by the catalog class
dfcat = dfcat.reset_index().rename(columns=column_name_mapper)
# create the origin_times from decimal years
dfcat['origin_time'] = dfcat.apply(lambda row: decimal_year_to_utc_epoch(row.year), axis=1)

# create catalog from dataframe
catalog = CSEPCatalog.from_dataframe(dfcat)
print(catalog)

#We need to link the region to catalog to forecast region.
catalog.region = forecast_gridded.region
####################################################################################################################################
# Compute Poisson spatial test and Number test
# ------------------------------------------------------
#
# Simply call the :func:`csep.core.poisson_evaluations.spatial_test`  and :func:`csep.core.poisson_evaluations.number_test` functions to evaluate the forecast using the specified
# evaluation catalog. The spatial test requires simulating from the Poisson forecast to provide uncertainty. The verbose
# option prints the status of the simulations to the standard output.

spatial_test_result = poisson.spatial_test(forecast_gridded, catalog)
number_test_result = poisson.number_test(forecast_gridded, catalog)


####################################################################################################################################
# Plot spatial test results
# -------------------------
#
# We provide the function :func:`csep.utils.plotting.plot_poisson_consistency_test` to visualize the evaluation results from
# consistency tests.

ax_spatial = plots.plot_poisson_consistency_test(spatial_test_result,
                                        plot_args={'xlabel': 'Spatial likelihood'})
ax_number = plots.plot_poisson_consistency_test(number_test_result,
                                        plot_args={'xlabel': 'Number of Earthquakes'})