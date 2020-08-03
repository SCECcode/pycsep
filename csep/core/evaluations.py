import numpy


class EvaluationResult:

    def __init__(self, test_distribution=None, name=None, observed_statistic=None, quantile=None, status="",
                       obs_catalog_repr='', sim_name=None, obs_name=None, min_mw=None):
        """
        Stores the result of an evaluation.

        Args:
            test_distribution (1d array-like): collection of statistics computed from stochastic event sets
            name (str): name of the evaluation
            observed_statistic (float or int): statistic computed from target catalog
            quantile (tuple or float): quantile of observed statistic from test distribution
            status (str): optional
            obs_catalog_repr (str): text information about the catalog used for the evaluation
            sim_name (str): name of simulation
            obs_name (str): name of observed catalog
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
            'min_mw': self.min_mw
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


