"""
This class is responsible for taking a workflow and preparing all of the
necessary components to complete the calculation.

The user specifies all machines and runtime configurations.

Experiment configurations are serialized into JSON to load for further processing.
"""
import os
import sys
import uuid
from copy import deepcopy
from csep.core.config import machine_config, repository_config
from csep.core.jobs import job_builder
from csep.core.system import system_builder
from csep.core.repositories import repo_builder
from csep.core.exceptions import CSEPSchedulerException

class Workflow:

    def __init__(self, config):
        self.name = config['name']
        self._defaults = {
            "system": None,
            "repository": None
        }
        self._jobs = []
        self._base_dir = config['base_dir']

    @property
    def base_dir(self):
        return self._base_dir

    def add_job(self, name, config):
        """
        Add computational job to the workflow.

        Args:
            job:

        Returns:

        """
        try:
            job = job_builder.create(name, config)
        except ValueError:
            print('Unable to build forecast job object.')
            sys.exit(-1)
        self._jobs.append(job)
        return job

    def prepare(self):
        """
        Creates computing environments for each job in workflow.

        Returns:
            None

        Throws:
            OSError
        """
        for job in self._jobs:
            job.prepare()

    def default_resource(self, name):
        """
        Resources corresponding to keys in the system configuration dictionary.


        Args:
            name (str): identifier for the system represented with the 'name' key in the config dict

        Returns:
            None
        """
        try:
            config = machine_config[name]
        except KeyError:
            print('Error. Machine configuration not registered with the program.')
            sys.exit(-1)

        try:
            system = system_builder.create(config['name'], config)
        except ValueError:
            print('Unable to build system configuration object.')
            sys.exit(-1)

        self._defaults['resource'] = system

    def default_repository(self, name, url=None):
        """
        Default repository for all Files in the workflow.

        Args:
            name (str): name corresponding to entry in repository configuration dict.

        Returns:

        """
        try:
            config = repository_config[name]
        except KeyError:
            print('Error. Repository not registered with the program.')

        try:
            repository = repo_builder.create(name)
        except ValueError:
            print('Unable to build repository object.')
            sys.exit(-1)

        self._defaults['repository'] = repository

    def to_dict(self):
        exclude = ['_jobs']
        out = {}
        for k, v in self.__dict__.items():
            if not callable(v) and v not in exclude:
                if hasattr(v, 'to_dict'):
                    new_v = v.to_dict()
                else:
                    new_v = str(v)
                if k.startswith('_'):
                    out[k[1:]] = new_v
                else:
                    out[k] = new_v

            # custom serializaing for jobs
            out["jobs"] = {}
            for job in self._jobs:
                out["jobs"][job.run_id]=job.to_dict()
        return out

    def archive(self):
        # this rolls through the repository layer
        pass


class ForecastExperiment(Workflow):
    """
    CSEP Specfic implementation for earthquake forecasting research.
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.name = config['name']

    def add_forecast(self, config, force=False):
        """
        Add forecast to experiment manager. Wraps around workflow add_job() to provide
        an interface that makes more sense in the context of earthquake forecasting experiments.

        Returns:

        """
        # need deepcopy to work on fresh copy of config file
        cfg=deepcopy(config)
        name=cfg['name']
        print(f'Adding {name} forecast to experiment.')
        # add unique working dir if not specified
        try:
            wd = cfg['work_dir']
        except KeyError:
            wd = None
            if self._base_dir is None:
                raise CSEPSchedulerException("Either need work_dir or base_dir set to add forecast to experiment.")
        if not wd:
            run_id = uuid.uuid4().hex
            work_dir = os.path.join(self._base_dir, run_id)
            cfg.update({'run_id': run_id, 'work_dir': work_dir})
        cfg.update({'force': force})
        job = self.add_job(name, cfg)
        return job
