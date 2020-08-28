.. _forecast-reference:

#########
Forecasts
#########

PyCSEP supports two types of earthquake forecasts that can be evaluated using the tools provided in this package.

1. Grid-based forecasts
2. Catalog-based forecasts

These forecast types and the PyCSEP objects used to represent them will be explained in detail in this document.

*****************
Gridded forecasts
*****************

Grid-based forecasts assume that earthquakes occur in independent and discrete space-time-magnitude bins. The occurrence
of these earthquakes are described only by their expected rates. This forecast format provides a general representation
of seismicity that can accommodate forecasts without explicit likelihood functions, such as those created using smoothed
seismicity models. Gridded forecasts can also be produced using simulation-based approaches like
epidemic-type aftershock sequence models.

Currently, grid-based forecasts define their spatial component using a 2D Cartesian (rectangular) grid, and
their magnitude bins using a 1D Cartesian (rectangular) grid. The last bin (largest magnitude) bin is assumed to
continue until infinity. Forecasts use latitude and longitude to define the bin edge of the spatial grid. Typical values
for the are 0.1° x 0.1° (lat x lon) and 0.1 ΔMw units. These choices are not strictly enforced and can defined
according the specifications of an experiment.

Working with gridded forecasts
##############################

PyCSEP provides the :class:`GriddedForecast<csep.core.forecasts.GriddedForecast>` class to handle working with
grid-based forecasts. This section will show aspects of this class.

.. autosummary:: csep.core.forecasts.GriddedForecast


***********************
Catalog-based forecasts
***********************

Catalog-based earthquake forecasts are issued as collections of synthetic earthquake catalogs. Every synthetic catalog
represents a realization of the forecast that is representative the uncertainty present in the model that generated
the forecast. Unlike grid-based forecasts, catalog-based forecasts retain the space-magnitude dependency of the events
they are trying to model. A grid-based forecast can be easily computed from a catalog-based forecast by assuming a
space-magnitude region and counting events within each bin from each catalog in the forecast. There can be issues with
under sampling, especially for larger magnitude events.


Working with catalog-based forecasts
####################################

.. autosummary:: csep.core.forecasts.CatalogForecast



