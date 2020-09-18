import json
import os
import time

# CSEP imports
from csep.models import EvaluationResult, CatalogNumberTestResult
from csep.core.forecasts import CatalogForecast, GriddedForecast
from csep.core.catalogs import UCERF3Catalog, CSEPCatalog
from csep.utils import readers
from csep.utils.time_utils import strptime_to_utc_datetime, utc_now_datetime


def load_stochastic_event_sets(filename, type='csv', format='native', **kwargs):
    """ General function to load stochastic event sets

    This function returns a generator to iterate through a collection of catalogs.
    To load a forecast and include metadata use :func:`csep.load_catalog_forecast`.

    Args:
        filename (str): name of file or directory where stochastic event sets live.
        type (str): either 'ucerf3' or 'csep' depending on the type of observed_catalog to load
        format (str): ('csep' or 'native') if native catalogs are not converted to csep format.
        kwargs (dict): see the documentation of that class corresponding to the type you selected
                         for the kwargs options

    Returns:
        (generator): :class:`~csep.core.catalogs.AbstractBaseCatalog`

    """
    if type not in ('ucerf3', 'csv'):
        raise ValueError("type must be one of the following: (ucerf3)")

    # use mapping to dispatch to correct function
    # in general, stochastic event sets are loaded with classmethods and single catalogs use the
    # constructor
    mapping = {'ucerf3': UCERF3Catalog.load_catalogs,
               'csv': CSEPCatalog.load_ascii_catalogs}

    # dispatch to proper loading function
    result = mapping[type](filename, **kwargs)

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



def load_catalog(filename, type='csep-csv', format='native', loader=None, **kwargs):
    """ General function to load single catalog

    See corresponding class documentation for additional parameters.

    Args:
        type (str): ('ucerf3', 'csep-csv', 'zmap', 'jma-csv', 'ndk') default is 'csep-csv'
        format (str): ('native', 'csep') determines whether the catalog should be converted into the csep
                      formatted catalog or kept as native.

    Returns (:class:`~csep.core.catalogs.AbstractBaseCatalog`)
    """

    if type not in ('ucerf3', 'csep-csv', 'zmap', 'jma-csv', 'ndk') and loader is None:
        raise ValueError("type must be one of the following: ('ucerf3', 'csep-csv', 'zmap', 'jma-csv', 'ndk').")

    # map to correct catalog class, at some point these could be abstracted into configuration file
    # this maps a human readable string to the correct catalog class and the correct loader function
    class_loader_mapping = {
        'ucerf3': {
            'class': UCERF3Catalog,
            'loader': None
        },
        'csep-csv': {
            'class': CSEPCatalog,
            'loader': readers.csep_ascii
        },
        'zmap': {
            'class': CSEPCatalog,
            'loader': readers.zmap_ascii
        },
        'jma-csv': {
            'class': CSEPCatalog,
            'loader': readers.jma_csv,
        },
        'ndk': {
            'class': CSEPCatalog,
            'loader': readers.ndk
        }
    }

    # treat json files using the from_dict() member instead of constructor
    catalog_class = class_loader_mapping[type]['class']
    if os.path.splitext(filename)[-1][1:] == 'json':
        catalog = catalog_class.load_json(filename, **kwargs)
    else:
        if loader is None:
            loader = class_loader_mapping[type]['loader']

        catalog = catalog_class.load_catalog(filename=filename, loader=loader, **kwargs)

    # convert to csep format if needed
    if format == 'native':
        return_val = catalog
    elif format == 'csep':
        return_val = catalog.get_csep_format()
    else:
        raise ValueError('format must be either "native" or "csep"')
    return return_val

def query_comcat(start_time, end_time, min_magnitude=2.50,
                 min_latitude=31.50, max_latitude=43.00,
                 min_longitude=-125.40, max_longitude=-113.10, verbose=True, **kwargs):
    """
    Access Comcat catalog through web service

    Args:
        start_time: datetime object of start of catalog
        end_time: datetime object for end of catalog
        min_magnitude: minimum magnitude to query
        min_latitude:  maximum magnitude to query
        max_latitude: max latitude of bounding box
        min_longitude: min latitude of bounding box
        max_longitude: max longitude of bounding box
        region: :class:`csep.core.regions.CartesianGrid2D
        verbose (bool): print catalog summary statistics

    Returns:
        :class:`csep.core.catalogs.ComcatCatalog
    """

    # Timezone should be in UTC
    t0 = time.time()
    eventlist = readers._query_comcat(start_time=start_time, end_time=end_time,
                           min_magnitude=min_magnitude,
                           min_latitude=min_latitude, max_latitude=max_latitude,
                           min_longitude=min_longitude, max_longitude=max_longitude)
    t1 = time.time()
    comcat = CSEPCatalog(data=eventlist, date_accessed=utc_now_datetime(), **kwargs)
    print("Fetched ComCat catalog in {} seconds.\n".format(t1 - t0))
    if verbose:
        print("Downloaded catalog from ComCat with following parameters")
        print("Start Date: {}\nEnd Date: {}".format(str(comcat.start_time), str(comcat.end_time)))
        print("Min Latitude: {} and Max Latitude: {}".format(comcat.min_latitude, comcat.max_latitude))
        print("Min Longitude: {} and Max Longitude: {}".format(comcat.min_longitude, comcat.max_longitude))
        print("Min Magnitude: {}".format(comcat.min_magnitude))
        print(f"Found {comcat.event_count} events in the ComCat catalog.")
    return comcat

def load_evaluation_result(fname):
    """ Load evaluation result stored as json file

    Returns:
        :class:`csep.core.evaluations.EvaluationResult`

    """
    # tries to return the correct class for the evaluation result. if it cannot find the type simply returns the basic result.
    evaluation_result_factory = {
        'default': EvaluationResult,
        'CatalogNumberTestResult': CatalogNumberTestResult
    }
    with open(fname, 'r') as json_file:
        json_dict = json.load(json_file)
        try:
            evaluation_type = json_dict['named_type']
        except:
            evaluation_type = 'default'
    eval_result = evaluation_result_factory[evaluation_type].from_dict(json_dict)
    return eval_result

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
        'dat': GriddedForecast.load_ascii,
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
    extension = os.path.splitext(fname)[-1][1:]
    if extension not in forecast_loader_mapping.keys() and loader is None:
        raise AttributeError("File extension should be in ('dat','xml','h5','bin') if loader not provided.")

    if extension in ('xml','h5','bin'):
        raise NotImplementedError

    # assign default loader
    if loader is None:
        loader = forecast_loader_mapping[extension]
    forecast = loader(fname, **kwargs)
    # final sanity check
    if not isinstance(forecast, GriddedForecast):
        raise ValueError("Forecast not instance of GriddedForecast")
    return forecast

def load_catalog_forecast(fname, catalog_loader=None, format='native', type='ascii', **kwargs):
    """ General function to handle loading catalog forecasts.

        Currently, just a simple wrapper, but can contain more complex logic in the future.

        Args:
            fname (str): pathname to the forecast file or directory containing the forecast files
            catalog_loader (func): callable that can load catalogs, see load_stochastic_event_sets above.
            format (str): either 'native' or 'csep'. if 'csep', will attempt to be returned into csep catalog format. used to convert between
                          observed_catalog type.
            type (str): either 'ucerf3' or 'csep', determines the catalog format of the forecast. if loader is provided, then
                        this parameter is ignored.
            **kwargs: other keyword arguments passed to the :class:`csep.core.forecasts.CatalogForecast`.

        Returns:
            :class:`csep.core.forecasts.CatalogForecast`
    """
    # sanity checks
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Could not locate file {fname}. Unable to load forecast.")
    # sanity checks
    if catalog_loader is not None and not callable(catalog_loader):
        raise AttributeError("Loader must be callable. Unable to load forecast.")
    # factory methods for loading different types of catalogs
    catalog_loader_mapping = {
        'ascii': CSEPCatalog.load_ascii_catalogs,
        'ucerf3': UCERF3Catalog.load_catalogs
    }
    if catalog_loader is None:
        catalog_loader = catalog_loader_mapping[type]
    # try and parse information from filename and send to forecast constructor
    if format == 'native' and type=='ascii':
        # this works for unix how windows?
        basename = str(os.path.basename(fname.rstrip('/')).split('.')[0])
        split_fname = basename.split('_')
        name = split_fname[0]
        start_time = strptime_to_utc_datetime(split_fname[1], format="%Y-%m-%dT%H-%M-%S-%f")
        # update kwargs
        _ = kwargs.setdefault('name', name)
        _ = kwargs.setdefault('start_time', start_time)
    # create observed_catalog forecast
    return CatalogForecast(filename=fname, loader=catalog_loader, catalog_format=format, catalog_type=type, **kwargs)