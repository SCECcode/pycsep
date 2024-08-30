"""
Plot customizations
===================

This example shows how to include some advanced options in the spatial visualization
of Gridded Forecasts and Evaluation Results

Overview:
    1. Define optional plotting arguments
    2. Set extent of maps
    3. Visualizing selected magnitude bins
    4. Plot global maps
    5. Plot multiple Evaluation Results

"""

################################################################################################################
# Example 1: Spatial dataset plot arguments
# -----------------------------------------

####################################################################################################################################
# **Load required libraries**

import csep
import cartopy
import numpy
from csep.utils import datasets, plots

import matplotlib.pyplot as plt

####################################################################################################################################
# **Load a Grid Forecast from the datasets**
#
forecast = csep.load_gridded_forecast(datasets.hires_ssm_italy_fname,
                                      name='Werner, et al (2010) Italy')
####################################################################################################################################
# **Plotting the dataset with fine-tuned arguments**

####################################################################################################################################
# These arguments are, in order:
#
# * Set an extent
# * Select ESRI Imagery as a basemap.
# * Assign a title
# * Set labels to the geographic axes
# * Draw country borders
# * Set a linewidth of 0.5 to country border
# * Assign ``'rainbow'`` as colormap. Possible values from ``matplotlib.cm`` library
# * Defines 0.8 for an exponential transparency function (default is 0 for constant alpha, whereas 1 for linear).
# * An object cartopy.crs.Projection() is passed as Projection to the map
#
# The complete description of plot arguments can be found in :func:`csep.utils.plots.plot_gridded_dataset`

####################################################################################################################################

ax = forecast.plot(extent=[3, 22, 35, 48],
                   basemap='ESRI_imagery',
                   title='Italy 10 year forecast',
                   grid_labels=True,
                   borders=True,
                   borders_linewidth=1.5,
                   cmap='rainbow',
                   alpha_exp=0.8,
                   projection=cartopy.crs.Mercator(),
                   show=True)

####################################################################################################################################
# Example 2: Plot a global forecast and a selected magnitude bin range
# --------------------------------------------------------------------
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
# We get the total rate between these magnitudes

rate_sum = rates_mw.sum(axis=1)

####################################################################################################################################
# The data is stored in a 1D array, so it should be projected into `region` 2D cartesian grid.

rate_sum = forecast.region.get_cartesian(rate_sum)

####################################################################################################################################
# **Plotting the dataset**
# To plot a global forecast, we must assign the option ``set_global=True``, which is required by :ref:cartopy to handle
# internally the extent of the plot. We can further customize the plot using the required arguments from :func:`csep.utils.plots.plot_gridded_dataset`

ax = plots.plot_gridded_dataset(numpy.log10(rate_sum), forecast.region,
                                figsize=(10,6),
                                set_global=True,
                                coastline_color='black',
                                projection=cartopy.crs.Robinson(central_longitude=180.0),
                                title=forecast.name,
                                grid_labels=False,
                                colormap='magma',
                                clabel= r'$\log_{10}\lambda\left(M_w \in [{%.2f},\,{%.2f}]\right)$ per ' \
                                        r'${%.1f}^\circ\times {%.1f}^\circ $ per forecast period' % \
                                        (low_bound, upper_bound, forecast.region.dh, forecast.region.dh),
                                show=True)

####################################################################################################################################



####################################################################################################################################
# Example 3: Plot a catalog
# -------------------------------------------

####################################################################################################################################
# **Load a Catalog from ComCat**

start_time = csep.utils.time_utils.strptime_to_utc_datetime('2002-01-01 00:00:00.0')
end_time = csep.utils.time_utils.strptime_to_utc_datetime('2012-01-01 00:00:00.0')
min_mag = 4.5
catalog = csep.query_comcat(start_time, end_time, min_magnitude=min_mag, verbose=False)

# **Define plotting arguments*

####################################################################################################################################
# These arguments are, in order:
#
# * Assign as basemap a TIFF file in the same reference as the plot
# * Set minimum markersize of 2 with red color
# * Set a 0.3 transparency
# * mag_scale is used to exponentially scale the size with respect to magnitude. Recommended 1-8
# * Set legend True and location in 3 (lower-left corner)
# * Set a list of Magnitude ticks to display in the legend
#
# The complete description of plot arguments can be found in :func:`csep.utils.plots.plot_catalog`
# The arguments can be stored as a dictionary a priori and then unpacked to the function with `**`.


plot_args = {'basemap': csep.datasets.basemap_california,
             'size': 7,
             'max_size': 500,
             'markercolor': 'red',
             'alpha': 0.3,
             'mag_scale': 8,
             'legend': True,
             'legend_loc': 3,
             'coastline': False,
             'mag_ticks': [4.0, 5.0, 6.0, 7.0]}

####################################################################################################################################

# **Plot the catalog**
ax = catalog.plot(show=True, **plot_args)


####################################################################################################################################
# Example 4: Plot multiple evaluation results
# -------------------------------------------

####################################################################################################################################
# Load L-test results from example .json files (See
# :doc:`gridded_forecast_evaluation` for information on calculating and storing evaluation results)

L_results = [csep.load_evaluation_result(i) for i in datasets.l_test_examples]
plot_args = {'figsize': (6, 5),
             'title': r'$\mathcal{L}-\mathrm{test}$',
             'title_fontsize': 18,
             'xlabel': 'Log-likelihood',
             'xticks_fontsize': 9,
             'ylabel_fontsize': 9,
             'linewidth': 0.8,
             'capsize': 3,
             'hbars':True,
             'tight_layout': True}

####################################################################################################################################
# Description of plot arguments can be found in :func:`plot_poisson_consistency_test`.
# We set ``one_sided_lower=True`` as usual for an L-test, where the model is rejected if the observed
# is located within the lower tail of the simulated distribution.
ax = plots.plot_consistency_test(L_results,
                                 one_sided_lower=True,
                                 show=True,
                                 **plot_args)



