API Reference
=============

This contains a reference document to the PyCSEP API.

.. automodule:: csep

.. :currentmodule:: csep

Loading catalogs, forecasts and results
---------------------------------------

.. autosummary::
   :toctree: generated

   load_catalog
   query_comcat
   query_bsi
   query_gns
   query_gcmt
   load_gridded_forecast
   load_catalog_forecast
   load_stochastic_event_sets
   load_evaluation_result

Catalogs
--------

.. :currentmodule:: csep.core.catalogs

.. automodule:: csep.core.catalogs


Catalog operations are defined using :class:`AbstractBaseCatalog` class.

.. autosummary::
   :toctree: generated

   AbstractBaseCatalog
   CSEPCatalog
   UCERF3Catalog

Catalog operations
------------------

Input and output operations for catalogs:

.. autosummary::
   :toctree: generated

   CSEPCatalog.to_dict
   CSEPCatalog.from_dict
   CSEPCatalog.to_dataframe
   CSEPCatalog.from_dataframe
   CSEPCatalog.write_json
   CSEPCatalog.load_json
   CSEPCatalog.load_catalog
   CSEPCatalog.write_ascii
   CSEPCatalog.load_ascii_catalogs
   CSEPCatalog.get_csep_format
   CSEPCatalog.plot

Accessing event information:

.. autosummary::
   :toctree: generated

   CSEPCatalog.event_count
   CSEPCatalog.get_magnitudes
   CSEPCatalog.get_longitudes
   CSEPCatalog.get_latitudes
   CSEPCatalog.get_depths
   CSEPCatalog.get_epoch_times
   CSEPCatalog.get_datetimes
   CSEPCatalog.get_cumulative_number_of_events

Filtering and binning:

.. autosummary::
   :toctree: generated

   CSEPCatalog.filter
   CSEPCatalog.filter_spatial
   CSEPCatalog.apply_mct
   CSEPCatalog.spatial_counts
   CSEPCatalog.magnitude_counts
   CSEPCatalog.spatial_magnitude_counts

Other utilities:

.. autosummary::
   :toctree: generated

   CSEPCatalog.update_catalog_stats
   CSEPCatalog.length_in_seconds
   CSEPCatalog.get_bvalue

.. currentmodule:: csep.core.forecasts
.. automodule:: csep.core.forecasts

Forecasts
---------

PyCSEP provides classes to interact with catalog and grid based Forecasts

.. autosummary::
   :toctree: generated

   GriddedForecast
   CatalogForecast

Gridded forecast methods:

.. autosummary::
   :toctree: generated

   GriddedForecast.data
   GriddedForecast.event_count
   GriddedForecast.sum
   GriddedForecast.magnitudes
   GriddedForecast.min_magnitude
   GriddedForecast.magnitude_counts
   GriddedForecast.spatial_counts
   GriddedForecast.get_latitudes
   GriddedForecast.get_longitudes
   GriddedForecast.get_magnitudes
   GriddedForecast.get_index_of
   GriddedForecast.get_magnitude_index
   GriddedForecast.load_ascii
   GriddedForecast.from_custom
   GriddedForecast.get_rates
   GriddedForecast.target_event_rates
   GriddedForecast.scale_to_test_date
   GriddedForecast.plot

Catalog forecast methods:

.. autosummary::
   :toctree: generated

   CatalogForecast.magnitudes
   CatalogForecast.min_magnitude
   CatalogForecast.spatial_counts
   CatalogForecast.magnitude_counts
   CatalogForecast.get_expected_rates
   CatalogForecast.get_dataframe
   CatalogForecast.write_ascii
   CatalogForecast.load_ascii

.. automodule:: csep.core.catalog_evaluations

Evaluations
-----------

PyCSEP provides implementations of evaluations for both catalog-based forecasts and grid-based forecasts.

Catalog-based forecast evaluations:

.. autosummary::
   :toctree: generated

   number_test
   spatial_test
   magnitude_test
   resampled_magnitude_test
   MLL_magnitude_test
   pseudolikelihood_test
   calibration_test

.. automodule:: csep.core.poisson_evaluations

Grid-based forecast evaluations:

.. autosummary::
   :toctree: generated

   number_test
   magnitude_test
   spatial_test
   likelihood_test
   conditional_likelihood_test
   paired_t_test
   w_test

Evaluation result base class

.. automodule:: csep.models

.. autosummary::
   :toctree: generated

    EvaluationResult


.. currentmodule:: csep.core.regions
.. automodule:: csep.core.regions

Regions
-------

PyCSEP includes commonly used CSEP testing regions and classes that facilitate working with gridded data sets. This
module is early in development and help is welcome here!

Region class(es):

.. autosummary::
    :toctree: generated

    CartesianGrid2D
    CartesianGrid2D.from_origins
    CartesianGrid2D.from_dict
    CartesianGrid2D.to_dict
    CartesianGrid2D.get_cartesian
    CartesianGrid2D.get_cell_area
    CartesianGrid2D.get_index_of

    QuadtreeGrid2D
    QuadtreeGrid2D.from_catalog
    QuadtreeGrid2D.from_single_resolution
    QuadtreeGrid2D.from_quadkeys
    QuadtreeGrid2D.get_cartesian

Testing regions:

.. autosummary::
    :toctree: generated

    california_relm_region
    california_relm_collection_region
    italy_csep_region
    italy_csep_collection_region
    nz_csep_region
    nz_csep_collection_region
    global_region

Region utilities:

.. autosummary::
    :toctree: generated

    magnitude_bins
    create_space_magnitude_region
    parse_csep_template
    increase_grid_resolution
    masked_region
    generate_aftershock_region

.. currentmodule:: csep.plots
.. automodule:: csep.plots

.. _api-plot-functions:

Plotting
--------

PyCSEP offers multiple plot functionalities to visually explore forecasts, catalogs and results.

Exploratory plotting:

.. autosummary::
   :toctree: generated

   plot_magnitude_versus_time
   plot_cumulative_events_versus_time
   plot_magnitude_histogram

Spatial plotting:

.. autosummary::
   :toctree: generated

   plot_basemap
   plot_catalog
   plot_gridded_dataset

Plotting grid-based evaluations:

.. autosummary::
   :toctree: generated

   plot_comparison_test
   plot_consistency_test

Plotting catalog-based evaluations:

.. autosummary::
   :toctree: generated

   plot_test_distribution
   plot_calibration_test

Plotting alarm-based evaluations:

.. autosummary::
   :toctree: generated

   plot_ROC_diagram
   plot_concentration_ROC_diagram
   plot_Molchan_diagram


.. automodule:: csep.utils.time_utils

Time Utilities
--------------

.. autosummary::
   :toctree: generated

   epoch_time_to_utc_datetime
   datetime_to_utc_epoch
   millis_to_days
   days_to_millis
   strptime_to_utc_epoch
   timedelta_from_years
   strptime_to_utc_datetime
   utc_now_datetime
   utc_now_epoch
   create_utc_datetime
   decimal_year

.. automodule:: csep.utils.comcat

Comcat Access
-------------

We integrated the code developed by Mike Hearne and others at the USGS to reduce the dependencies of this package. We plan
to move this to an external and optional dependency in the future.

.. autosummary::
   :toctree: generated

   search
   get_event_by_id

.. automodule:: csep.utils.calc

Calculation Utilities
---------------------

.. autosummary::
   :toctree: generated

   nearest_index
   find_nearest
   func_inverse
   discretize
   bin1d_vec

.. automodule:: csep.utils.stats

Statistics Utilities
--------------------

.. autosummary::
   :toctree: generated

   sup_dist
   sup_dist_na
   cumulative_square_diff
   binned_ecdf
   ecdf
   greater_equal_ecdf
   less_equal_ecdf
   min_or_none
   max_or_none
   get_quantiles
   poisson_log_likelihood
   poisson_joint_log_likelihood_ndarray
   poisson_inverse_cdf

.. automodule:: csep.utils.basic_types

Basic types
-----------

.. autosummary::
    :toctree: generated

    AdaptiveHistogram