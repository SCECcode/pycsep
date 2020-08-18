.. _catalogs-reference:

########
Catalogs
########

PyCSEP provides routines for working with and manipulating earthquake catalogs for the purposes of evaluating earthquake
forecasting models.

If you are able to make use of these tools for other reasons, please let us know! We are especially interested in
including basic catalog statistics into this package. If you are interested in helping implement routines like
b-value estimation and catalog completeness please let us know!

.. contents:: Table of Contents
    :local:
    :depth: 2

************
Introduction
************

PyCSEP catalog basics
=====================

An earthquake catalog contains a collection of seismic events each defined by a set of attributes. PyCSEP implements
a simple event description that is suitable for evaluating earthquake forecasts. In this format, every seismic event is
defined by its location in space (longitude, latitude, and depth), magnitude, and origin time. In addition, each event
can have an optional ``event_id`` as a unique identifier.

PyCSEP provides :class:`csep.core.catalogs.CSEPCatalog` to represent an earthquake catalog. The essential event data are stored in a
`structured NumPy array <https://numpy.org/doc/stable/user/basics.rec.html>`_ with the following data type. ::

    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<i8'),
                         ('latitude', '<f4'),
                         ('longitude', '<f4'),
                         ('depth', '<f4'),
                         ('magnitude', '<f4')])

Additional information can be associated with an event using the ``id`` field in the structured array in a class member called
``metadata``. The essential event data must be complete meaning that each event should have these attributes defined. The metadata
is more freeform and PyCSEP does not impose any restrictions on the way that event metadata is stored. Only that the metadata
for an event should be accessible using the event ``id``. An example of this could be ::

    catalog = csep.load_catalog('catalog_file.csv')
    event_metadata = catalog.metadata[event_id]

PyCSEP contains catalog readers for the following formats (We are also looking to support other catalog formats. Please
suggest some!):

1. CSEP ascii format
2. NDK format (used by the gCMT catalog)
3. INGV gCMT catalog
4. ZMAP format
5. pre-processed JMA format

PyCSEP supports the ability to easily define a custom reader function for a catalog format type that we don't currently support.
If you happen to implement a reader for a new catalog format please check out the `contribution guidelines <https://github.com/SCECCode/csep2/blob/dev/CONTRIBUTING.md>`_
and make a pull request so we can include this in the next release.
