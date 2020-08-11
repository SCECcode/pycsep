Developer Notes
===============

Last updated: 10 August 2020

Catalogs
--------

Right now the catalogs are implemented on top of :class:`csep.core.catalogs.AbstractBaseCatalog`. This was originally done
to accommodate the custom format from UCERF3-ETAS and provide web access to ComCat. It became clear that dealing with multiple
catalog classes will become to cumbersome, so I'm proposing that we write a single catalog class with multiple readers that can
load data into the appropriate format. Some considerations are that we might want to accommodate more data than is available through the
regular CSEP format (ie, time, lon, lat, depth, magnitude, event_id, catalog_id).

The current principle is that when a catalog has a different data model (eg, UCERF3-ETAS including fields like fss_index and
erf_index) it would become its own catalog class. This is defined from the catalog dtype (eg., UCERF3Catalog.dtype).
Right now, we have three catalogs with effectively the same dtype: (1) CSEPCatalog, ComcatCatalog, ZMAPCatalog, and the JmaCsvCatalog.
The ZMAPCatalog is slightly different, because it just expands the date into multiple columns.

We have options that should be decided on before release:
(1) Maintain the current design and allow the catalogs to be subclassed freely
(2) Remove the similar catalogs (above) and replace them with different readers (eg, read_ndk). These functions would
    read various file-formats (and access web-portals) and return a tuple of the event definitions in the appropriate format
    for the CSEP catalog.
(3) A combination of items (1) and (2) where most catalog formats will be read into the "CSEP" format, but still provide the
    ability to subclass the catalog if required.


Reproducibility Files
---------------------

Store information for reproducibility. This should include the following:
    1. version information (git hash of commit)
    2. forecast filename
    3. evaluation catalog (including necessary information to recreate the filtering properties); maybe just md5
    4. do we need calculation dates?

Evaluation Results
------------------

* Each evaluation should return an evaluation result class that has an associated .plot() method.
* We should be able to .plot() most everything that makes sense including forecasts and evaluation results.

    * How do we .plot() a catalog?
    * Should we .plot() a region?
    * For serialization, we can identify the appropriate class as a string in the class state and use that to create the correct object on load.

Forecast metadata information
-----------------------------

1. Forecast should contain metadata information to identify properties of the forecast

    * Start and end date
    * Spatial region
    * Magnitude bins

Working with GriddedForecasts
-----------------------------

* Right now, we can only spatial counts over the entire magnitude range. What if we wanted to have some control over this?
* Might want to plot above some magnitude threshold or within some incremental threshold.
* Should be able to have a method that returns a new GriddedForecast with specified parameters such as min/max magnitude.

Region information
------------------
* The region information will need to accommodate more complex spaces including 3D areas and those with non-regular grids (e.g.,
  quadtrees or meshes)


