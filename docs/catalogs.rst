Catalogs
========

CSEP2 relaxes the assumption that forecasts supply Poissonian rates on structured and regular grids by allowing forecasts to
supply :ref:`stochastic event sets <stochastic-event-set>`. Classes will be defined for each catalog type and should extend the
:class:`BaseCatalog <csep.core.catalogs.BaseCatalog>` class.

.. autoclass:: csep.core.catalogs.BaseCatalog
  :members:

.. autoclass:: csep.core.catalogs.CSEPCatalog
  :members:

.. autoclass:: csep.core.catalogs.UCERF3Catalog
  :members:

.. autoclass:: csep.core.catalogs.ComcatCatalog
  :members:
