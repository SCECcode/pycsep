import numpy
from csep.utils.stats import get_quantiles, binned_ecdf, sup_dist
from csep.utils import flat_map_to_ndarray


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

        """
        creates evaluation result from a dictionary
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


def _distribution_test(stochastic_event_set_data, observation_data):

    # for cached files want to write this with memmap
    union_catalog = flat_map_to_ndarray(stochastic_event_set_data)
    min_time = 0.0
    max_time = numpy.max([numpy.max(numpy.ceil(union_catalog)), numpy.max(numpy.ceil(observation_data))])

    # build test_distribution with 100 data points
    num_points = 100
    tms = numpy.linspace(min_time, max_time, num_points, endpoint=True)

    # get combined ecdf and obs ecdf
    combined_ecdf = binned_ecdf(union_catalog, tms)
    obs_ecdf = binned_ecdf(observation_data, tms)

    # build test distribution
    n_cat = len(stochastic_event_set_data)
    test_distribution = []
    for i in range(n_cat):
        test_ecdf = binned_ecdf(stochastic_event_set_data[i], tms)
        # indicates there were zero events in catalog
        if test_ecdf is not None:
            d = sup_dist(test_ecdf[1], combined_ecdf[1])
            test_distribution.append(d)
    d_obs = sup_dist(obs_ecdf[1], combined_ecdf[1])

    # score evaluation
    _, quantile = get_quantiles(test_distribution, d_obs)

    return test_distribution, d_obs, quantile

def _compute_likelihood_old(gridded_data, apprx_rate_density, expected_cond_count, n_obs):
    """
    not sure if this should actually be used masked arrays, bc we are losing information about undetermined l-test results.
    apply spatial smoothing here?
    """
    gridded_cat_ma = numpy.ma.masked_where(gridded_data == 0, gridded_data)
    apprx_rate_density_ma = numpy.ma.array(apprx_rate_density, mask=gridded_cat_ma.mask)
    likelihood = numpy.sum(gridded_cat_ma * numpy.log10(apprx_rate_density_ma)) - expected_cond_count
    # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio, this should never be 0, else forecast is expecting 0 earthquakes in region.
    normalizing_factor = n_obs / expected_cond_count
    normed_rate_density_ma = normalizing_factor * apprx_rate_density_ma
    # compute likelihood for each event, ignoring cells with 0 events in the catalog.
    likelihood_norm = numpy.ma.sum(gridded_cat_ma * numpy.ma.log10(normed_rate_density_ma)) / numpy.ma.sum(gridded_cat_ma)
    return (likelihood, likelihood_norm)

def _compute_likelihood(gridded_data, apprx_rate_density, expected_cond_count, n_obs):
    # compute pseudo likelihood
    idx = gridded_data != 0

    # this value is: -inf obs at idx and no apprx_rate_density
    #                -expected_cond_count if no target earthquakes
    likelihood = numpy.sum(gridded_data[idx] * numpy.log10(apprx_rate_density[idx])) - expected_cond_count

    # comes from Eq. 20 in Zechar et al., 2010., normalizing forecast by event count ratio.
    normalizing_factor = n_obs / expected_cond_count
    n_cat = numpy.sum(gridded_data)
    norm_apprx_rate_density = apprx_rate_density * normalizing_factor

    # value could be: -inf if no value in apprx_rate_dens
    #                  nan if n_cat is 0 and above condition holds
    #                  inf if n_cat is 0
    likelihood_norm = numpy.sum(gridded_data[idx] * numpy.log10(norm_apprx_rate_density[idx])) / n_cat

    return (likelihood, likelihood_norm)

def _compute_spatial_statistic(gridded_data, log10_probability_map):
    """
    aggregates the log1
    Args:
        gridded_data:
        log10_probability_map:
    """
    # returns a unique set of indexes corresponding to cells where earthquakes occurred
    idx = numpy.unique(numpy.argwhere(gridded_data))
    return numpy.sum(log10_probability_map[idx])

def number_test(stochastic_event_sets, observation, event_counts=None):
    # get number of events for observations and simulations
    if not event_counts:
        sim_counts = []
        for catalog in stochastic_event_sets:
            sim_counts.append(catalog.event_count)
    else:
        sim_counts = event_counts
    observation_count = observation.event_count
    # get delta_1 and delta_2 values
    delta_1, delta_2 = get_quantiles(sim_counts, observation_count)
    # prepare result
    result = EvaluationResult(test_distribution=sim_counts,
                              name='N-Test',
                              observed_statistic=observation_count,
                              quantile=(delta_1, delta_2),
                              status='Normal',
                              obs_catalog_repr=str(observation),
                              sim_name=stochastic_event_sets[0].name,
                              obs_name=observation.name)
    return result




