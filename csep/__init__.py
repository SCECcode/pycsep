import os
import time
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import UCERF3Catalog, ZMAPCatalog, ComcatCatalog
from csep.core.evaluations import EvaluationResult
from csep.core.repositories import FileSystem
from csep.utils.file import get_file_extension
from csep.core.exceptions import CSEPIOException

# change type to catalog_
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
    if type not in ('ucerf3',):
        raise ValueError("type must be one of the following: (ucerf3)")

    # use mapping to dispatch to correct function
    # in general, stochastic event sets are loaded with classmethods and single catalogs use the
    # constructor
    mapping = {'ucerf3': UCERF3Catalog.load_catalogs}

    # dispatch to proper loading function
    result = mapping[type](**kwargs)

    # factory function to load catalogs from different classes
    while True:
        try:
            catalog = next(result)
        except StopIteration:
            return
        except Exception:
            raise
        if format == 'native':
            yield catalog
        elif format == 'csep':
            yield catalog.get_csep_format()
        else:
            raise ValueError('format must be either "native" or "csep!')


def load_catalog(filename, type=None, format='native', **kwargs):
    """ Method to load single catalog. See corresponding class documentation for additional parameters.

    Args:
        type (str): either 'ucerf3' or 'comcat'
        format (str): ('csep' or 'native'), and determines whether the catalog should be converted into the csep
                      formatted catalog or kept as native.

    Returns (:class:`~csep.core.catalogs.AbstractBaseCatalog`)
    """
    if type not in ('ucerf3', 'comcat', 'csep'):
        raise ValueError("type must be one of the following: ('ucerf3', 'comcat', 'csep')")

    # add entry point to load catalog here.
    mapping = {'ucerf3': UCERF3Catalog,
               'csep': ZMAPCatalog,
               'comcat': ComcatCatalog}

    # treat json files using the from_dict() member instead of constructor
    if get_file_extension(filename) == 'json':
        catalog = mapping[type](filename=None, **kwargs)
        repo = FileSystem(url=filename)
        catalog = repo.load(catalog)
    else:
        catalog = mapping[type](filename=filename, **kwargs)

    # convert to csep format if needed
    if format == 'native':
        return_val = catalog
    elif format == 'csep':
        return_val = catalog.get_csep_format()
    else:
        raise ValueError('format must be either "native" or "csep"')
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

def load_evaluation_result(fname):
    """ Load evaluation result stored as json file.

    Returns:
        :class:`csep.core.evaluations.EvaluationResult`

    """
    repo = FileSystem(url=fname)
    result = repo.load(EvaluationResult())
    return result

def load_gridded_forecast(fname, loader=None, **kwargs):
    """ Loads grid based forecast from hard-disk.

    The function loads the forecast provided with at the filepath defined by fname. The function attempts to understand
    the file format based on the extension of the filepath. Optionally, if loader function is provided, that function
    will be used to load the forecast. The loader function should return a :class:`csep.core.forecasts.GriddedForecast`
    class with the region and magnitude members correctly assigned.

    File extensions:
        .dat -> CSEP ascii format
        .xml -> CSEP xml format (not yet implemented)
        .h5 -> CSEP hdf5 format (not yet implemented)
        .bin -> CSEP binary format (not yet implemented)

    Args:
        fname (str): path of grid based forecast
        loader (func): function to load forecast in bespoke format needs to return :class:`csep.core.forecasts.GriddedForecast`
                       and first argument should be required and the filename of the forecast to load
                       called as loader(func, **kwargs).

        **kwargs: passed into loader function

    Throws:
        FileNotFoundError: when the file extension is not known and a loader is not provided.
        AttributeError: if loader is provided and is not callable.

    Returns:
        :class:`csep.core.forecasts.GriddedForecast`
    """
    # mapping from file extension to loader function, new formats by default they need to be added here
    forecast_loader_mapping = {
        'dat': GriddedForecast.from_csep1_ascii,
        'xml': None,
        'h5': None,
        'bin': None
    }
    # sanity checks
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Could not locate file {fname}. Unable to load forecast.")
    # sanity checks
    if loader is not None and not callable(loader):
        raise AttributeError("Loader must be callable. Unable to load forecast.")
    extension = get_file_extension(fname)
    if extension not in forecast_loader_mapping.keys() and loader is None:
        raise AttributeError("File extension should be in ('dat','xml','h5','bin') if loader not provided.")
    # assign default loader
    if loader is None:
        loader = forecast_loader_mapping[extension]
    forecast = loader(fname, **kwargs)
    # final sanity check
    if not isinstance(forecast, GriddedForecast):
        raise ValueError("Forecast not instance of GriddedForecast")
    return forecast




