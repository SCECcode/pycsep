.. _regions-reference

#######
Regions
#######

.. automodule:: csep.utils.basic_types

PyCSEP includes commonly used CSEP testing regions and classes that facilitate working with gridded data sets. This
module is early in development and will be a focus of future development.

.. contents:: Table of Contents
    :local:
    :depth: 2

.. :currentmodule:: csep

.. automodule:: csep.core.regions

Practically speaking, earthquake forecasts, especially time-dependent forecasts, treat time differently than space and
magnitude. If we consider a family of monthly forecasts for the state of California for earthquakes with **M** 3.95+,
each of these forecasts would use the same space-magnitude region, even though the time periods are
different. Because the time horizon is an implicit property of the forecast, we do not explicitly consider time in the region
objects provided by PyCSEP. This module contains tools for working with gridded regions in both space and magnitude.

First, we will describe how the spatial regions are handled. Followed by magnitude regions, and how these two aspects
interact with one another.

**************
Region objects
**************

Currently, PyCSEP provides two different kinds of spatial gridding approaches to handle binning catalogs and defining regions 
for earthquake forecasting evaluations, i.e. :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` and :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>`.
The fruther details about spatial grids are given below.

2D Cartesian grids
##################

This section contains information about using 2D cartesian grids.

.. autosummary::

    CartesianGrid2D

.. note::
    We are planning to do some improvements to this module and to expand its capabilities. For example, we would like to
    handle non-regular grids such as a quad-tree. Also, a single Polygon should be able to act as the spatial component
    of the region. These additions will make this toolkit more useful for crafting bespoke experiments and for general
    catalog analysis. Feature requests are always welcome!

The :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` acts as a data structure that can associate a spatial
location (eg., lon and lat) with its corresponding spatial bin. This class is optimized to work with regular grids,
although they do not have to be complete (they can have holes) and they do not have to be rectangular (each row / column
can have a different starting coordinate).

The :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` maintains a list of
:class:`Polygon<csep.core.regions.Polygon>` objects that represent the individual spatial bins from the overall
region. The origin of each polygon is considered to be the lower-left corner (the minimum latitude and minimum longitude).

.. autosummary::

    CartesianGrid2D.num_nodes
    CartesianGrid2D.get_index_of
    CartesianGrid2D.get_location_of
    CartesianGrid2D.get_masked
    CartesianGrid2D.get_cartesian
    CartesianGrid2D.get_bbox
    CartesianGrid2D.midpoints
    CartesianGrid2D.origins
    CartesianGrid2D.from_origins


Creating spatial regions
########################

Here, we describe how the class works starting with the class constructors. ::

    @classmethod
    def from_origins(cls, origins, dh=None, magnitudes=None, name=None):
        """ Convenience function to create CartesianGrid2D from list of polygon origins """

For most applications, using the :meth:`from_origins<csep.core.regions.CartesianGrid2D.from_origins>` function will be
the easiest way to create a new spatial region. The method accepts a 2D :class:`numpy.ndarray` containing the x (lon) and y (lat)
origins of the spatial bin polygons. These should be the complete set of origins. The function will attempt to compute the
grid spacing by comparing the x and y values between adjacent origins. If this does not seem like a reliable approach
for your region, you can explicitly provide the grid spacing (dh) to this method.

When a :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` is created the following steps occur:

    1. Compute the bounding box containing all polygons (2D array)
    2. Create a map between the index of the 2D bounding box and the list of polygons of the region.
    3. Store a boolean flag indicating whether a given cell in the 2D array is valid or not

Once these mapping have been created, we can now associate an arbitrary (lon, lat) point with a spatial cell using the
mapping defined in (2). The :meth:`get_index_of<csep.core.regions.CartesianGrid2D.get_index_of>` accepts a list
of longitudes and latitudes and returns the index of the polygon they are associated with. For instance, this index can
now be used to access a data value stored in another data structure.

***************
Testing Regions
***************

CSEP has defined testing regions that can be used for earthquake forecasting experiments. The following functions in the
:mod:`csep.core.regions` module returns a :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` consistent with
these regions.

.. autosummary::

    california_relm_region
    italy_csep_region
    global_region

****************
Region Utilities
****************

PyCSEP also provides some utilities that can facilitate working with regions. As we expand this module, we will include
functions to accommodate different use-cases.

.. autosummary::

    magnitude_bins
    create_space_magnitude_region
    parse_csep_template
    increase_grid_resolution
    masked_region
    generate_aftershock_region
	

Quadtree based grid
##################

The other gridding approach PyCSEP supports is quadtree-based spatial gridding approach. 
The quadtree is a hierarchical tiling strategy for storing and indexing geospatial data. In start whole globe is divided into 4 tiles,  identified as '0', '1', '2', '3'.  
Then each tile can be divided into four children tiles, until a final desired grid is acquired.
Each tile is identified by a unique identifier called quadkey, which are '0', '1', '2' or '3'. When a tile is divided further, the quadkey is also modified by appending the new number with the previous number. 
The number of times a tile is divided is reffered as the zoom-level. And the length of quadkey denotes the number of times a tile has been divided. 
If a single-resolution grid is acquired at zoom-level (L) or 5, it gives 1024 spatial cells in whole globe. Similary at L=11, the number of cells acquired in grid are 4.2 million approx. 
The quadtree-based gridding approach that is also made capable of providing a multi-resolution spatial grid in addition to single-resolution grid.
In a multi-resolution grid, the grid resolution is determined by a certain criterion (or multiple criteria), e.g. maximum number of earthquakes allowed per cell (Nmax). 
This means that only those cells (tiles) are divided further into sub-cells that contain more earthquakes than Nmax. 
Thus, quadtree approach can be employed to generate high-resolution (smaller) grid cells in seismically active regions and a low-resolution (bigger) grid cells in seismically quiet regions. 
It offers earthquake forecast modellers the liberty of choosing a suitable spatial grid based on the dataset available for training of forecast models.


This section contains information about using quadtree based grid.

.. autosummary::

    QuadtreeGrid2D


The :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>` acts as a data structure that can associate a spatial
location, identified by a quadkey (or lon and lat) with its corresponding spatial bin. This class allows to create a quadtree grid using three different methods, based on the choice of user. 
It also offers the conversion from quadtree cell to lon/lat bouds.

The :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>` maintains a list of
:class:`Polygon<csep.core.regions.Polygon>` objects that represent the individual spatial bins from the overall
region. The origin of each polygon is considered to be the lower-left corner (the minimum latitude and minimum longitude).

.. autosummary::

    QuadtreeGrid2D.num_nodes
	QuadtreeGrid2D.compute_cell_area
    QuadtreeGrid2D.get_index_of
    QuadtreeGrid2D.get_location_of
    QuadtreeGrid2D.get_bbox
    QuadtreeGrid2D.midpoints
    QuadtreeGrid2D.origins
	QuadtreeGrid2D.save_quadtree
    QuadtreeGrid2D.from_catalog
	QuadtreeGrid2D.from_regular_grid
	QuadtreeGrid2D.from_quadkeys


Creating spatial regions
########################

Here, we describe how the quadtree grid is created using class methods in three different ways as required by the users.

Catalog-based multi-resolution grid
-----------------------------------------
Read a global earthquake catalog in :class:`CSEPCatalog<csep.core.catalogs.CSEPCatalog>` format and use it to generate a multi-resolution quadtree-based grid.
 
.. code-block:: default

    #Get global catalog in the format of csep.core.catalogs.CSEPCatalog
    #Define allowed maximum number of earthquakes per cell (Nmax)
    Nmax = 10
    r = QuadtreeGrid2D.from_catalog(CSEPCatalog, Nmax)
	#saving quadtree grid
	r.save_quadtree(filename)

Quadtree single-resolution grid
-----------------------------------------
Generate a grid at the same zoom-level everywhere. This grid does not require a catalog. It only needs the zoom-level a which the single-resolution grid is required.

.. code-block:: default
	zoom_level = 11
	r = QuadtreeGrid2D.from_regular_grid(zoom_level)
	#saving quadtree grid
	r.save_quadtree(filename)
	
Quadtree grid loading from file
---------------------------------------
An already saved quadtree grid can also be loaded in the pyCSEP.

..code-block:: default
	qk = numpy.genfromtxt(filename, dtype = 'str')
    r = QuadtreeGrid2D.from_quadkeys(qk) 

When a :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>` is created the following steps occur:

    1. Compute the bounding box containing all polygons (2D array)
    2. Create a map between the index of the 2D bounding box and the list of polygons of the region.
    3. Translate all the quatree cells in corresponding longitude/latitude bounds just as additional information and use.

Once these mapping have been created, we can now associate an arbitrary (lon, lat) point with a spatial cell using the
mapping defined in (2). The :meth:`get_index_of<csep.core.regions.QuadtreeGrid2D.get_index_of>` accepts a list
of longitudes and latitudes and returns the index of the polygon they are associated with. For instance, this index can
now be used to access a data value stored in another data structure.

***************
Testing Regions
***************

CSEP has defined testing regions that can be used for earthquake forecasting experiments. So far quadtree-based grids are tested for global forecast regions. 
However, a quadtree based region can be acquired for any geographical area and used for forecast generation and testing. 

