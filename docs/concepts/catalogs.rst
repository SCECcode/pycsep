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

An earthquake catalog contains a collection of seismic events defined by their attributes. PyCSEP implements
a simple event description that is suitable for evaluating earthquake forecasts. In this format, every seismic event is
defined by its location in space (longitude, latitude, and depth), magnitude, and origin time. In addition, each event
can have an optional ``event_id`` as a unique identifier.

PyCSEP provides :class:`csep.core.catalogs.Catalog` to represent an earthquake catalog. The essential event data are stored in a
`structured NumPy array <https://numpy.org/doc/stable/user/basics.rec.html>`_ with the following data type.::

    dtype = numpy.dtype([('id', 'S256'),
                         ('origin_time', '<i8'),
                         ('latitude', '<f4'),
                         ('longitude', '<f4'),
                         ('depth', '<f4'),
                         ('magnitude', '<f4')])

Additional information can be associated with an event using the ``id`` field in the structured array. By default, the CSEP
formats do not include additional metadata, but this could be an option.








*****************************
Implementing a catalog reader
*****************************

