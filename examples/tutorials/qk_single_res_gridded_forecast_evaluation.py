"""

.. _quadtree_single_res-forecast-evaluation:

Quadtree Single-Resolution Grid-based Forecast Evaluation
=======================================

This example demonstrates how to evaluate a single-resolution quadtree grid-based time-independent forecast. Grid-based
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
from csep.utils import datasets, time_utils, plots
from csep.core.regions import QuadtreeGrid2D
from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year_to_utc_epoch
from csep.core.catalogs import CSEPCatalog

####################################################################################################################################
# Define Gridded Region
# --------------------------
#
# Here as an example we define a single resolution grid at zoom-level L=6. For this purpose 
# we call :func:`csep.core.regions.QuadtreeGrid2D_from_single_resolution` to create a single resolution grid.

# For simplicity of example, we assume only single magnitude bin, 
# i.e. all the earthquakes greater than and equal to 5.95

mbins = numpy.array([5.95])
r = QuadtreeGrid2D.from_single_resolution(6, magnitudes=mbins)
print('Number of cells in grid :', r.num_nodes)

####################################################################################################################################
# Load forecast
# -------------
# We have already created a time-independent global forecast with time horizon of 1 year and provided with the reporsitory.   
# The filepath is relative to the root directory of the package. You can specify any file location for your forecasts.

forecast_data = numpy.loadtxt('example_rate_zoom=6.csv')
#Reshape forecast as Nx1 array
forecast_data = forecast_data.reshape(-1,1)

forecast_gridded = GriddedForecast(data = forecast_data, region = r, 
                                   magnitudes = mbins, name = 'Example Single-res Forecast')

#The loaded forecast is for 1 year. The test catalog we will use is for 6 years. So we can rescale the forecast.
print(f"expected event count before scaling: {forecast_gridded.event_count}")
forecast_gridded.scale(6)
print(f"expected event count after scaling: {forecast_gridded.event_count}")
    

####################################################################################################################################
# Load evaluation catalog
# -----------------------
#
# We have a test catalog stored here in a different format as compare to the format required by PyCSEP. 
# So we can read the test catalog as a pandas frame and convert it into the format that is acceptable to PyCSEP.
# Then we instantiate an object of class CSEPCatalog by calling function :func:`csep.core.regions.CSEPCatalog.from_dataframe`

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
# Compute Poisson spatial test
# ----------------------------
#
# Simply call the :func:`csep.core.poisson_evaluations.spatial_test` and :func:`csep.core.poisson_evaluations.number_test` functions to evaluate the forecast using the specified
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
