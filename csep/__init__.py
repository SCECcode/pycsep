import time
from csep.core.catalogs import UCERF3Catalog, ZMAPCatalog, ComcatCatalog

def load_stochastic_event_sets(type=None, format='native', **kwargs):
    """
    Factory function to load stochastic event sets.
    # IDEA: should this return a stochastic event set class with a consistent api to apply things to an event set?

    Args:
        type (str): either 'ucerf3' or 'csep' depending on the type of catalog to load
        format (str): ('csep' or 'native') if native catalogs are not converted to csep format.
        **kwargs: see the documentation of that class corresponding to the type you selected

    Returns:
        (generator): :class:`~csep.core.catalogs.AbstractBaseCatalog`

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
    while True:
        try:
            catalog = next(result)
        except StopIteration:
            raise
        except Exception:
            raise
        if format == 'native':
            yield(catalog)
        elif format == 'csep':
            yield(catalog.get_csep_format())
        else:
            raise ValueError('format must be either "native" or "csep!')


def load_catalog(type=None, format='native', **kwargs):
    """
    Factory function to load single catalog. See corresponding class documentation for additional parameters.

    Args:
        type (str): either 'ucerf3' or 'comcat'
        format (str): ('csep' or 'native')

    Returns:
        (:class:`~csep.core.catalogs.ZMAPCatalog`)
    """

    if type not in ('ucerf3', 'comcat', 'csep'):
        raise ValueError("type must be one of the following: ('ucerf3', 'comcat', 'csep')")

    # add entry point to load catalog here.
    mapping = {'ucerf3': UCERF3Catalog,
               'csep': ZMAPCatalog,
               'comcat': ComcatCatalog}

    # load catalog in native format
    catalog = mapping[type](**kwargs)

    # convert to csep format
    if format == 'native':
        return_val = catalog
    elif format == 'csep':
        return_val = catalog.get_csep_format()
    else:
        raise ValueError('format must be either "native" or "csep!')
    return return_val


def load_comcat(start_time, end_time, min_magnitude=2.50,
                min_latitude=31.50, max_latitude=43.00,
                min_longitude=-125.40, max_longitude=-113.10, region=None, verbose=True):

    # Todo: check if datetime has utc timezone and if not set it, and issue a warning to the user.

    t0 = time.time()
    comcat = ComcatCatalog(start_time=start_time, end_time=end_time,
                           name='Comcat', min_magnitude=min_magnitude,
                           min_latitude=min_latitude, max_latitude=max_latitude,
                           min_longitude=min_longitude, max_longitude=max_longitude, region=region, query=True)
    t1 = time.time()
    print("Fetched Comcat catalog in {} seconds.\n".format(t1 - t0))
    if verbose:
        print("Downloaded Comcat Catalog with following parameters")
        print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
        print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
        print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
        print("Min Magnitude: {}".format(comcat.min_magnitude))
        print(f"Found {comcat.get_number_of_events()} events in the Comcat catalog.")
        print(f'Proceesing Catalogs.')
    return comcat