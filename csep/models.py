import numpy

# CSEP Imports
from csep.utils.time_utils import datetime_to_utc_epoch, epoch_time_to_utc_datetime
from csep.utils import plots


class Simulation:
    """
    View of CSEP Experiment. Contains minimal information required to perform evaluations of
    CSEP Forecasts
    """
    def __init__(self, filename='', min_mw=2.5, start_time=-1, sim_type='', name=''):
        self.filename = filename
        self.min_mw = min_mw
        self.start_time = start_time
        self.sim_type = sim_type
        self.name = name


class Event:
    def __init__(self, id=None, magnitude=None, latitude=None, longitude=None, time=None):
        self.id = id
        self.magnitude = magnitude
        self.latitude = latitude
        self.longitude = longitude
        self.time = time

    @classmethod
    def from_dict(cls, adict):
        return cls(id=adict['id'],
                  magnitude=adict['magnitude'],
                  latitude=adict['latitude'],
                  longitude=adict['longitude'],
                  time=epoch_time_to_utc_datetime(adict['time']))


    def to_dict(self):
        adict = {
            'id': self.id,
            'magnitude': self.magnitude,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'time': datetime_to_utc_epoch(self.time)
        }
        return adict


class EvaluationResult:

    def __init__(self, test_distribution=None, name=None, observed_statistic=None, quantile=None, status="",
                       obs_catalog_repr='', sim_name=None, obs_name=None, min_mw=None):
        """
        Stores the result of an evaluation.

        Args:
            test_distribution (1d array-like): collection of statistics computed from stochastic event sets
            name (str): name of the evaluation
            observed_statistic (float or int): statistic computed from target observed_catalog
            quantile (tuple or float): quantile of observed statistic from test distribution
            status (str): optional
            obs_catalog_repr (str): text information about the observed_catalog used for the evaluation
            sim_name (str): name of simulation
            obs_name (str): name of observed observed_catalog
        """
        self.test_distribution=test_distribution
        self.name = name
        self.observed_statistic = observed_statistic
        self.quantile = quantile
        self.status = status
        self.obs_catalog_repr = obs_catalog_repr
        self.sim_name = sim_name
        self.obs_name = obs_name
        self.min_mw = min_mw
        # this will be used for object creation
        self.named_type = self.__class__.__name__

    def to_dict(self):
        try:
            td_list = self.test_distribution.tolist()
        except AttributeError:
            td_list = list(self.test_distribution)
        adict = {
            'name': self.name,
            'sim_name': self.sim_name,
            'obs_name': self.obs_name,
            'obs_catalog_repr': self.obs_catalog_repr,
            'quantile': self.quantile,
            'observed_statistic': self.observed_statistic,
            'test_distribution': td_list,
            'status': self.status,
            'min_mw': self.min_mw,
            'type': self.named_type
        }
        return adict

    @classmethod
    def from_dict(cls, adict):

        """ Creates evaluation result from a dictionary
        Args:
            adict (dict): stores information about classes

        Returns:

        """
        new = cls(test_distribution=numpy.array(adict['test_distribution']),
            name=adict['name'],
            observed_statistic=adict['observed_statistic'],
            quantile=adict['quantile'],
            sim_name=adict['sim_name'],
            obs_name=adict['obs_name'],
            obs_catalog_repr=adict['obs_catalog_repr'],
            status=adict['status'],
            min_mw=adict['min_mw'])
        return new

    def plot(self):
        raise NotImplementedError("plot not implemented on EvaluationResult class.")


class CatalogNumberTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, show=False, plot_args=None):
        plot_args = plot_args or {}
        td = self.test_distribution
        min_bin, max_bin = numpy.min(td), numpy.max(td)
        # hard-code some logic for bin size
        bins = numpy.arange(min_bin, max_bin)
        if len(bins) == 1:
            bins = 3
        # compute bin counts, this one is special because of integer values
        plot_args_defaults = {'percentile': 95,
                              'title': f'Number Test',
                              'xlabel': 'Event count in catalog',
                              'bins': bins}
        # looks funny, but will update the defaults with the user defined arguments
        plot_args_defaults.update(plot_args)
        ax = plots.plot_number_test(self, show=show, plot_args=plot_args)
        return ax

class CatalogPseudolikelihoodTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, show=False, plot_args=None):
        plot_args = plot_args or {}
        # compute bin counts, this one is special because of integer values
        plot_args_defaults = {'percentile': 95,
                              'title': 'Pseudolikelihood Test',
                              'bins': 'auto'}
        # looks funny, but will update the defaults with the user defined arguments
        plot_args_defaults.update(plot_args)
        ax = plots.plot_likelihood_test(self, show=show, plot_args=plot_args)
        return ax

class CatalogMagnitudeTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, show=False, plot_args=None):
        plot_args = plot_args or {}
        plot_args_defaults = {'percentile': 95,
                              'title': 'Magnitude Test',
                              'bins': 'auto'}
        plot_args_defaults.update(plot_args)
        ax = plots.plot_magnitude_test(self, show=show, plot_args=plot_args)
        return ax

class CatalogSpatialTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, show=False, plot_args=None):
        plot_args = plot_args or {}
        # compute bin counts, this one is special because of integer values
        plot_args_defaults = {
            'percentile': 95,
            'title': f'Spatial Test',
            'bins': 'auto'
        }
        # looks funny, but will update the defaults with the user defined arguments
        plot_args_defaults.update(plot_args)
        ax = plots.plot_spatial_test(self, show=show, plot_args=plot_args)
        return ax

class CalibrationTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, show=False, axes=None, plot_args=None):
        plot_args = plot_args or {}
        # set plotting defaults
        plot_args_defaults = {
            'label': self.sim_name,
            'title': self.name
        }
        plot_args_defaults.update(plot_args)
        ax = plots.plot_calibration_test(self, show=show, axes=axes, plot_args=plot_args)
        return ax

class EvaluationConfiguration:
    """
    Store information about the evaluation which will be used to store metadata about the evaluation.
    """
    def __init__(self, compute_time=None, catalog_file=None, forecast_file=None, n_cat=None,
                 eval_start_epoch=None, eval_end_epoch=None, git_hash=None, evaluations=None, forecast_name=None):
        """
        Constructor for EvaluationConfiguration object

        Args:
            compute_time (int): utc_epoch_time in millis indicating time plotting was completed
            catalog_file (str): filename of the catalog used to evaluate forecast
            forecast_file (str): filename of the forecast
            n_cat (int): number of catalogs processed
            eval_start_epoch (int): utc_epoch_time indicating start time of evaluations
            eval_end_epoch (int): utc_epoch_time indiciating end time of evaluations
            git_hash (str): hash indicating commit used for evaluations
            evaluations (dict): version information about evaluations
        """
        self.compute_time = compute_time
        self.catalog_file = catalog_file
        self.forecast_file = forecast_file
        self.forecast_name = forecast_name
        self.n_cat = n_cat
        self.eval_start_epoch = eval_start_epoch
        self.eval_end_epoch = eval_end_epoch
        self.git_hash = git_hash
        self.evaluations = evaluations or []

    def to_dict(self):
        adict = {
            'compute_time': self.compute_time,
            'forecast_file': self.forecast_file,
            'catalog_file': self.catalog_file,
            'n_cat': self.n_cat,
            'forecast_name': self.forecast_name,
            'eval_start_epoch': self.eval_start_epoch,
            'eval_end_epoch': self.eval_end_epoch,
            'git_hash': self.git_hash,
            'evaluations': self.evaluations
        }
        return adict

    @classmethod
    def from_dict(cls, adict):
        new = cls( compute_time=adict['compute_time'],
             catalog_file=adict['catalog_file'],
             forecast_file=adict['forecast_file'],
             forecast_name=adict['forecast_name'],
             n_cat=adict['n_cat'],
             eval_start_epoch=adict['eval_start_epoch'],
             eval_end_epoch=adict['eval_end_epoch'],
             git_hash=adict['git_hash'],
             evaluations=adict['evaluations'])
        return new

    def get_evaluation_version(self, name):
        for e in self.evaluations:
            if e['name'] == name:
                return e['version']
        return None

    def get_fnames(self, name):
        for e in self.evaluations:
            if e['name'] == name:
                return e['fnames']
        return None

    def update_version(self, name, version, fnames):
        found = False
        for e in self.evaluations:
            if e['name'] == name:
                e['version'] = version
                e['fnames'] = fnames
                found = True
        if not found:
            self.evaluations.append({'name': name, 'version': version, 'fnames': fnames})