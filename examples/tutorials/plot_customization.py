"""
Plotting customization
=========================

This example shows how to include some advanced options in the spatial visualization
of Gridded Forecasts and its Evaluation Results

Overview:
    1. Define optional plotting arguments
    2. Set extent of maps
    3. Visualizing selected magnitude bins
    4. Plot global maps
    5. Plot multiple Evaluation Results

"""

####################################################################################################################################
# Example 1: Spatial dataset plot arguments
# -----------------------------------------

####################################################################################################################################
# **Load required libraries**

import csep
import cartopy
import numpy
from csep.utils import datasets, plots

####################################################################################################################################
# **Load a Grid Forecast from the datasets**
#
forecast = csep.load_gridded_forecast(datasets.hires_ssm_italy_fname,
                                      name='Werner, et al (2010) Italy')
####################################################################################################################################
# **Selecting plotting arguments**
#
# Create a dictionary containing the plot arguments
args_dict = {'title': 'Italy 10 year forecast',
             'grid_labels': True,
             'borders': True,
             'feature_lw': 0.5,
             'basemap': 'ESRI_imagery',
             'cmap': 'rainbow',
             'alpha_exp': 0.8,
             'projection': cartopy.crs.Mercator()}
####################################################################################################################################
# These arguments are, in order:
#
# * Assign a title
# * Set labels to the geographic axes
# * Draw country borders
# * Set a linewidth of 0.5 to country borders
# * Select ESRI Imagery as a basemap.
# * Assign 'rainbow' as a colormap
# * Defines a 0.8 exponent for the transparency function.
# * An object cartopy.crs.Projection() is passed as Projection to the map
#
# The complete description of plot arguments can be found in :func:`csep.utils.plots.plot_spatial_dataset`

####################################################################################################################################
# **Plotting the dataset**
#
# The map `extent` can be defined. Otherwise, the extent of the data would be used. The dictionary defined must be passed as argument

ax = forecast.plot(extent=[3, 22, 35, 48],
                   show=True,
                   plot_args=args_dict)

####################################################################################################################################
# Example 2: Global forecast and selected magnitude bin range
# -----------------------------------------------------------
#
#
# **Load a Global Forecast from the datasets**
#
# A downsampled version of the `GEAR1 <http://peterbird.name/publications/2015_GEAR1/2015_GEAR1.htm>`_ forecast can be found in datasets.

forecast = csep.load_gridded_forecast(datasets.gear1_downsampled_fname,
                                      name='GEAR1 Forecast (downsampled)')

####################################################################################################################################
# **Filter by magnitudes**
#
# We get the rate of events of 5.95<=M_w<=7.5

low_bound = 6.15
upper_bound = 7.55
mw_bins = forecast.get_magnitudes()
mw_ind = numpy.where(numpy.logical_and( mw_bins >= low_bound, mw_bins <= upper_bound))[0]
rates_mw = forecast.data[:, mw_ind]

####################################################################################################################################
# We get the cumulative rate

cumulative_rate = rates_mw.sum(axis=1)

####################################################################################################################################
# The data comes in a 1D array. It should be placed into the `region` 2D cartesian grid

cumulative_rate_cartesian = forecast.region.get_cartesian(cumulative_rate)

####################################################################################################################################
# **Define plot arguments**
#
# We define the arguments and a global projection, centered at $lon=-180$

plot_args={'figsize': (10,6), 'coastline':True, 'feature_color':'black',
          'projection': cartopy.crs.Robinson(central_longitude=-180.0),
          'title': forecast.name, 'grid_labels': False,
          'cmap': 'magma',
          'clabel': r'$\log_{10}\lambda({%.2f}\leq M_w\leq{%.2f})$ per'
                    r'${%.1f}^\circ\times {%.1f}^\circ $ per forecast period' %
                    (low_bound, upper_bound, forecast.region.dh, forecast.region.dh)}

####################################################################################################################################
# **Plotting the dataset**
#

ax = plots.plot_spatial_dataset(numpy.log10(cumulative_rate_cartesian), forecast.region,
                                show=True, set_global=True,
                                plot_args=plot_args)

####################################################################################################################################
# Note that we have selected ``set_global=True``
