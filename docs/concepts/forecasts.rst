.. _forecast-reference:

#########
Forecasts
#########

pyCSEP supports two types of earthquake forecasts that can be evaluated using the tools provided in this package.

1. Grid-based forecasts
2. Catalog-based forecasts

These forecast types and the pyCSEP objects used to represent them will be explained in detail in this document.

.. contents:: Table of Contents
    :local:
    :depth: 2

*****************
Gridded forecasts
*****************

Grid-based forecasts assume that earthquakes occur in independent and discrete space-time-magnitude bins. The occurrence
of these earthquakes are described only by their expected rates. This forecast format provides a general representation
of seismicity that can accommodate forecasts without explicit likelihood functions, such as those created using smoothed
seismicity models. Gridded forecasts can also be produced using simulation-based approaches like
epidemic-type aftershock sequence models.

Currently, pyCSEP offers support for two types of grid-baesd forecasts, i.e. conventional gridded forecasts and quadtree-based gridded forecasts.
Conventional grid-based forecasts define their spatial component using a 2D Cartesian (rectangular) grid, and
their magnitude bins using a 1D Cartesian (rectangular) grid. The last bin (largest magnitude) bin is assumed to
continue until infinity. Forecasts use latitude and longitude to define the bin edge of the spatial grid. Typical values
for the are 0.1° x 0.1° (lat x lon) and 0.1 ΔMw units. These choices are not strictly enforced and can defined
according the specifications of an experiment.

pyCSEP aso offers support to handle forecast using quadtree approach. Single or multi-resolution spatial grid can be generated based on the choice of modelers.
Then that grid can used for generating earthquake forecast. 


Working with conventional gridded forecasts
###########################################

PyCSEP provides the :class:`GriddedForecast<csep.core.forecasts.GriddedForecast>` class to handle working with
grid-based forecasts. Please see visit :ref:`this example<grid-forecast-evaluation>` for an end-to-end tutorial on
how to evaluate a grid-based earthquake forecast.

.. autosummary:: csep.core.forecasts.GriddedForecast

Default file format
--------------------

The default file format of a gridded-forecast is a tab delimited ASCII file with the following columns
(names are not included): ::

    LON_0 	LON_1 	LAT_0 	LAT_1 	DEPTH_0 DEPTH_1 MAG_0 	MAG_1 	RATE					FLAG
    -125.4	-125.3	40.1	40.2	0.0     30.0	4.95	5.05	5.8499099999999998e-04	1

Each row represents a single space-magnitude bin and the entire forecast file contains the rate for a specified
time-horizon. An example of a gridded forecast for the RELM testing region can be found
`here <https://github.com/SCECcode/csep2/blob/dev/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj.aftershock-fromXML.dat>`_.


The coordinates (LON, LAT, DEPTH, MAG) describe the independent space-magnitude region of the forecast. The lower
coordinates are inclusive and the upper coordinates are exclusive. Rates are incremental within the magnitude range
defined by [MAG_0, MAG_1). The FLAG is a legacy value from CSEP testing centers that indicates whether a spatial cell should
be considered by the forecast. Currently, the implementation does not allow for individual space-magnitude cells to be
flagged. Thus, if a spatial cell is flagged then all corresponding magnitude cells are flagged.

.. note::
    PyCSEP only supports regions that have a thickness of one layer. In the future, we plan to support more complex regions
    including those that are defined using multiple depth regions. Multiple depth layers can be collapsed into a single
    layer by summing. This operations does reduce the resolution of the forecast.

Custom file format
------------------

The :meth:`GriddedForecast.from_custom<csep.core.forecasts.GriddedForecast.from_custom>` method allows you to provide
a function that can read custom formats. This can be helpful, because writing this function might be required to convert
the forecast into the appropriate format in the first place. This function has no requirements except that it returns the
expected data.

.. automethod:: csep.core.forecasts.GriddedForecast.from_custom


Working with quadtree-gridded forecasts
##############################################

The same forecast :class:`GriddedForecast<csep.core.forecasts.GriddedForecast>` class also handles forecasts with
quadtree grids. Please see visit :ref:`this example<grid-forecast-evaluation>` for an end-to-end tutorial on
how to evaluate a grid-based earthquake forecast.

.. autosummary:: csep.core.forecasts.GriddedForecast

Default file format
--------------------

The default file format of a quadtree gridded-forecast is also a tab delimited ASCII file with the following columns. Just one additional column is added to the file format, i.e. quadkey to identify the spatial cells.
If quadkeys for each spatial cell are known, it is enough to compute lon/lat bounds. However, lon/lat bounds are still kept in the default format to make it look consistent with conventional forecast format. 

(names are not included): ::

    QUADKEY	LON_0 	LON_1 	LAT_0 	LAT_1 	DEPTH_0 DEPTH_1 MAG_0 	MAG_1 	RATE					FLAG
    '01001'			-125.4	-125.3	40.1	40.2	0.0     30.0	4.95	5.05	5.8499099999999998e-04	1

Each row represents a single space-magnitude bin and the entire forecast file contains the rate for a specified
time-horizon.

The coordinates (LON, LAT, DEPTH, MAG) describe the independent space-magnitude region of the forecast. The lower
coordinates are inclusive and the upper coordinates are exclusive. Rates are incremental within the magnitude range
defined by [MAG_0, MAG_1). The FLAG is a legacy value from CSEP testing centers that indicates whether a spatial cell should
be considered by the forecast. Please note that flagged functionality is not yet included for quadtree-gridded forecasts.

PyCSEP offers the :func:`load_quadtree_forecast<csep.utils.readers.load_quadtree_forecast>` function to read quadtree forecast in default format.
Similarly, custom forecast can be defined and read into pyCSEP as explained for conventional gridded forecast.


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

Please see visit :ref:`this<catalog-forecast-evaluation>` example for an end-to-end tutorial on how to evaluate a catalog-based
earthquake forecast. An example of a catalog-based forecast stored in the default pyCSEP format can be found
`here <https://github.com/SCECcode/pycsep/blob/dev/csep/artifacts/ExampleForecasts/CatalogForecasts/ucerf3-landers_1992-06-28T11-57-34-14.csv>`_.


The standard format for catalog-based forecasts a comma separated value ASCII format. This format was chosen to be
human-readable and easy to implement in all programming languages. Information about the format is shown below.

.. note::
    Custom formats can be supported by writing a custom function or sub-classing the
    :ref:`AbstractBaseCatalog<csep.core.forecasts.AbstractBaseCatalog>`.

The event format matches the follow specfication: ::

    LON, LAT, MAG, ORIGIN_TIME, DEPTH, CATALOG_ID, EVENT_ID
    -125.4, 40.1, 3.96, 1992-01-05T0:40:3.1, 8, 0, 0

Each row in the catalog corresponds to an event. The catalogs are expected to be placed into the same file and are
differentiated through their `catalog_id`. Catalogs with no events can be handled in a couple different ways intended to
save storage.

The events within a catalog should be sorted in time, and the *catalog_id* should be increasing sequentially. Breaks in
the *catalog_id* are interpreted as missing catalogs.

The following two examples show how you represent a forecast with 5 catalogs each containing zero events.

**1. Including all events (verbose)** ::

    LON, LAT, MAG, ORIGIN_TIME, DEPTH, CATALOG_ID, EVENT_ID
    ,,,,,0,
    ,,,,,1,
    ,,,,,2,
    ,,,,,3,
    ,,,,,4,

**2. Short-hand** ::

    LON, LAT, MAG, ORIGIN_TIME, DEPTH, CATALOG_ID, EVENT_ID
    ,,,,,4,

The following three example show how you could represent a forecast with 5 catalogs. Four of the catalogs contain zero events
and one catalog contains one event.

**3. Including all events (verbose)** ::

    LON, LAT, MAG, ORIGIN_TIME, DEPTH, CATALOG_ID, EVENT_ID
    ,,,,,0,
    ,,,,,1,
    ,,,,,2,
    ,,,,,3,
    -125.4, 40.1, 3.96, 1992-01-05T0:40:3.1, 8, 4, 0

**4. Short-hand** ::

    LON, LAT, MAG, ORIGIN_TIME, DEPTH, CATALOG_ID, EVENT_ID
    -125.4, 40.1, 3.96, 1992-01-05T0:40:3.1, 8, 4, 0

The simplest way to orient the file follow (3) in the case where some catalogs contain zero events. The zero oriented
catalog_id should be assigned to correspond with the total number of catalogs in the forecast. In the case where every catalog
contains zero forecasted events, you would specify the forecasting using (2). The *catalog_id* should be assigned to
correspond with the total number of catalogs in the forecast.

