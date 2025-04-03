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
objects provided by pyCSEP. This module contains tools for working with gridded regions in both space and magnitude.

First, we will describe how the spatial regions are handled. Followed by magnitude regions, and how these two aspects
interact with one another.

.. **************
.. Region objects
.. **************

Currently, pyCSEP provides two different kinds of spatial gridding approaches to handle binning catalogs and defining regions
for earthquake forecasting evaluations, i.e. :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` and :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>`.
The fruther details about spatial grids are given below.

**************
Cartesian grid
**************

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

.. ***************
Testing Regions
########################

CSEP has defined testing regions that can be used for earthquake forecasting experiments. The following functions in the
:mod:`csep.core.regions` module returns a :class:`CartesianGrid2D<csep.core.regions.CartesianGrid2D>` consistent with
these regions.

.. autosummary::

    california_relm_region
    italy_csep_region
    global_region

.. ****************
Region Utilities
########################

PyCSEP also provides some utilities that can facilitate working with regions. As we expand this module, we will include
functions to accommodate different use-cases.

.. autosummary::

    magnitude_bins
    create_space_magnitude_region
    parse_csep_template
    increase_grid_resolution
    masked_region
    generate_aftershock_region

	
**************
Quadtree grid
**************

We want to use gridded regions with less spatial cells and multi-resolutions grids for creating earthquake forecast models.
We also want to test forecast models on different resolutions. But before we can do this, we need to have the capability to acquire such grids.  
There can be different possible options for creating multi-resolutions grids, such as voronoi cells or coarse grids, etc.
The gridding approach needs to have certain properties before we choose it for CSEP experiments. We want an approach for gridding that is simple to implement, easy to understand and should come with intutive indexing. Most importantly, it should come with a coordinated mechanism for changing between different resolutions. It means that one can not simply choose to combine cells of its own choice and create a larger grid cell (low-resolution) and vice versa. This can potentially make the grid comparision process difficult. There must be a specific well-defined strategy to change between different resolutions of grids. We explored different gridding approaches and found quadtree to be a better solution for this task, despite a few drawbacks, such as quadtree does not work for global region beyond 85.05 degrees North and South.   

The quadtree is a hierarchical tiling strategy for storing and indexing geospatial data. In start the global testing region is divided into 4 tiles,  identified as '0', '1', '2', '3'.  
Then each tile can be divided into four children tiles, until a final desired grid is acquired.
Each tile is identified by a unique identifier called quadkey, which are '0', '1', '2' or '3'. When a tile is divided further, the quadkey is also modified by appending the new identifier with the previous quadkey. 
Once a grid is acquired then we call each tile as a grid cell.
The number of times a tile is divided is reffered as the zoom-level (L) and the length of quadkey denotes the number of times a tile has been divided. 
If a grid has same zoom-level for each tile, then it is referred as single-resolution grid. 


A single-resolution grid is acquired at zoom-level L=5, provides 1024 spatial cells in whole globe. Increase in the value of L by one step leads to the increase in the number of grid cells by four times.
Similary at L=11, the number of cells acquired in grid are 4.2 million approximately.
 
We can use quadtree in combination with any data to create a multi-resolution grid, in which the resolution is determined by the input data. In general quadtree can be used in combination with any type of input data. However, now we provide support of earthquake catalog to be used as input data for determining the grid-resolution. With time we intend to incorporate the support for other types of datasets, such as such as distance form the mainshock or rupture plance, etc.
 
Currently, for generating mult-resolution grids, we can choose two criteria to decide resolution, i.e. maximum number of earthquakes allowed per cell (Nmax) and maximum zoom-level (L) allowed for a cell.
This means that only those cells (tiles) will be divided further into sub-cells that contain more earthquakes than Nmax and the cells will not be divided further after reaching L, even if number of earthquakes are more than Nmax. Thus, quadtree can provide high-resolution (smaller) grid cells in seismically active regions and a low-resolution (bigger) grid cells in seismically quiet regions. 
It offers earthquake forecast modellers the liberty of choosing a suitable spatial grid based on their choice.


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
    QuadtreeGrid2D.get_cell_area
    QuadtreeGrid2D.get_index_of
    QuadtreeGrid2D.get_location_of
    QuadtreeGrid2D.get_bbox
    QuadtreeGrid2D.midpoints
    QuadtreeGrid2D.origins
    QuadtreeGrid2D.save_quadtree
    QuadtreeGrid2D.from_catalog
    QuadtreeGrid2D.from_single_resolution
    QuadtreeGrid2D.from_quadkeys


Creating spatial regions
########################

Here, we describe how the class works starting with the class constructors and how users can create different types regions.

Multi-resolution grid based on earthquake catalog
-----------------------------------------
Read a global earthquake catalog in :class:`CSEPCatalog<csep.core.catalogs.CSEPCatalog>` format and use it to generate a multi-resolution quadtree-based grid constructors. ::

    @classmethod
    def from_catalog(cls, catalog, threshold, zoom=11, magnitudes=None, name=None):
        """ Convenience function to create a multi-resolution grid using earthquake catalog """

Single-resolution grid
-----------------------------------------
Generate a single-resolution grid at the same zoom-level everywhere. This grid does not require a catalog. It only needs the zoom-level to determine the resolution of grid.::

    @classmethod
    def from_single_resolution(cls, zoom, magnitudes=None, name=None):
        """ Convenience function to create a single-resolution grid """
	
Grid loading from already created quadkeys
---------------------------------------
An already saved quadtree grid can also be loaded in the pyCSEP. Read the quadkeys and use the following function to instantiate the class. ::

    @classmethod
    def from_quadkeys(cls, quadk, magnitudes=None, name=None):
        """ Convenience function to create a grid using already generated quadkeys """


When a :class:`QuadtreeGrid2D<csep.core.regions.QuadtreeGrid2D>` is created the following steps occur:

    1. Compute the bounding box containing all polygons (2D array) corresponding to quadkeys
    2. Create a map between the index of the 2D bounding box and the list of polygons of the region.


Once these mapping have been created, we can now associate an arbitrary (lon, lat) point with a spatial cell using the
mapping defined in (2). The :meth:`get_index_of<csep.core.regions.QuadtreeGrid2D.get_index_of>` accepts a list
of longitudes and latitudes and returns the index of the polygon they are associated with. For instance, this index can
now be used to access a data value stored in another data structure.


Testing Regions
########################

CSEP has defined testing regions that can be used for earthquake forecasting experiments. The above mentioned functions are used to create quadtree grids for global testing region.
Once a grid  
However, a quadtree gridded region can be acquired for any geographical area and used for forecast generation and testing. For example, we have created a quadtree-gridded region at fixed zoom-level of 12 for California RELM testing region.

.. autosummary::

    california_quadtree_region
