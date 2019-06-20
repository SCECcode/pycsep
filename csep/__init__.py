import logging
from csep.core.catalogs import UCERF3Catalog, CSEPCatalog, ComcatCatalog

# Configure only in your main program clause
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s %(levelname)s %(message)s')


def load_stochastic_event_set(type=None, format='native', **kwargs):
    """
    Factory function to load stochastic event sets.
    # IDEA: should this return a stochastic event set class with a consistent api to apply things to an event set?

    Args:
        type (str): either 'ucerf3' or 'csep' depending on the type of catalog to load
        format (str): ('csep' or 'native') if native catalogs are not converted to csep format.
        **kwargs: see the documentation of that class corresponding to the type you selected

    Returns:
        (generator): :class:`~csep.core.catalogs.BaseCatalog`

    """
    if type not in ('ucerf3'):
        raise ValueError("type must be one of the following: (ucerf3)")

    # use mapping to dispatch to correct function
    # in general, stochastic event sets are loaded with classmethods and single catalogs use the
    # constructor
    mapping = {'ucerf3': UCERF3Catalog.load_catalogs}

    # dispatch to proper loading function
    result = mapping[type](**kwargs)

    # convert to csep format
    for catalog in result:
        if format == 'native':
            yield(catalog)
        elif format == 'csep':
            yield(catalog._get_csep_format())
        else:
            raise ValueError('format must be either "native" or "csep!')


def load_catalog(type=None, format='native', **kwargs):
    """
    Factory function to load single catalog. See corresponding class documentation for additional parameters.

    Args:
        type (str): either 'ucerf3' or 'comcat'
        format (str): ('csep' or 'native')

    Returns:
        (:class:`~csep.core.catalogs.CSEPCatalog`)
    """

    if type not in ('ucerf3', 'comcat', 'csep'):
        raise ValueError("type must be one of the following: ('ucerf3', 'comcat', 'csep')")

    # add entry point to load catalog here.
    mapping = {'ucerf3': UCERF3Catalog,
               'csep': CSEPCatalog,
               'comcat': ComcatCatalog}

    # load catalog in native format
    catalog = mapping[type](**kwargs)

    # convert to csep format
    if format == 'native':
        return_val = catalog
    elif format == 'csep':
        return_val = catalog._get_csep_format()
    else:
        raise ValueError('format must be either "native" or "csep!')
    return return_val


class CSEPObject:
    """
    All Domain object should inherit this class.

    Implementing the to_dict() and from_dict() methods allows the class to be
    handled by the repository layer.
    """
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, adict):
        return cls(**adict)
