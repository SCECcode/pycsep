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
from csep.core.config import machine_config
from csep.core.jobs import job_builder
from csep.core.system import system_builder
from csep.core.repositories import repo_builder
from csep.core.exceptions import CSEPSchedulerException

class Workflow:
    """
    Top-level class for a computational workflow. This class is responsible for managing the state of the entire workflow.
    """

    def __init__(self, name='Unnamed', base_dir=None, default_system=None,
                 default_repository=None, owner=None, description=None):
        self.name = name
        self._defaults = {
            "system": default_system,
            "repository": default_repository
        }
        self.base_dir = base_dir
        self.owner = owner
        self.description=description
        self._repo = None
        self._system = None
        self._jobs = []

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

    def prepare(self, archive=False, dry_run=False):
        """
        Creates computing environments for each job in workflow.

        Returns:
            None

        Throws:
            OSError
        """
        for job in self._jobs:
            job.prepare(dry_run=dry_run)

        if archive:
            self.archive()

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
            self._system = system
        except ValueError:
            print('Unable to build system configuration object.')
            sys.exit(-1)

        self._defaults['system'] = name

    def add_repository(self, config):
        """
        Default repository for all Files in the workflow.

        Args:
            name (str): name corresponding to entry in repository configuration dict.

        Returns:

        """
        try:
            name = config['name']
            repository = repo_builder.create(name, **config)
            self._repo = repository
        except ValueError:
            raise CSEPSchedulerException("Unable to build repository object.")
            sys.exit(-1)
        self._defaults['repository'] = name

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
            # custom implementation for jobs
            out["jobs"] = {}
            for job in self._jobs:
                out["jobs"][job.run_id]=job.to_dict()
        return out

    def archive(self):
        """
        Will store the state of an experiment to the repository defined using the default_repository.


        The repository layer is only used for storing the state of an object. For example, files generated during
        a job will not be saved using the repository layer. These files are typically written out by 3rd party
        programs.

        Returns:
            None

        """
        if self._repo is None:
            print("Unable to archive simulation manifest. Repository must not be None.")
            return

        if not self._repo:
            print("Unable to access repository. Defaulting to FileSystem repository and storing in the experiment directory.")
            repo = repo_builder.create("filesystem", url=self.work_dir)

        else:
            repo = self._repo
            print(f"Found repository. Using {repo.name} to store class state.")

        # access storage through the repository layer
        # for sqlalchemy, this would create the Base objects to insert into the database.
        repo.save(self.to_dict())

    def run(self):
        # for now jobs should be responsible for running themselves
        # this represents a convenience wrapper. the manager should be uses for
        # monitoring jobs.
        for job in self._jobs:
            job.run()


class ForecastExperiment(Workflow):
    """
    CSEP Specfic implementation for earthquake forecasting research.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            if self.base_dir is None:
                raise CSEPSchedulerException("Either need work_dir or base_dir set to add forecast to experiment.")
        if not wd:
            run_id = uuid.uuid4().hex
            work_dir = os.path.join(self.base_dir, run_id)
            cfg.update({'run_id': run_id, 'work_dir': work_dir})
        cfg.update({'force': force})
        job = self.add_job(name, cfg)
        return job
