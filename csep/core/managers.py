"""
This class is responsible for taking a workflow and preparing all of the
necessary components to complete the calculation.

The user specifies all machines and runtime configurations.

Experiment configurations are serialized into JSON to load for further processing.
"""
import datetime
import os
import uuid
from copy import deepcopy

from csep.core.config import machine_config
from csep.core.jobs import job_builder
from csep.core.system import system_builder
from csep.core.repositories import repo_builder, Repository
from csep.core.exceptions import CSEPSchedulerException
from csep.utils.log import LoggingMixin

class Workflow(LoggingMixin):
    """
    Top-level class for a computational workflow. This class is responsible for managing the state of the entire workflow.
    """

    def __init__(self, name='Unnamed', base_dir='', owner=(), description='', repository=None):
        self.name = name
        self.base_dir = base_dir
        self.owner = owner
        self.description = description
        self.repository = repository
        if not isinstance(self.repository, Repository):
            if type(self.repository) is dict:
                self.repository = repo_builder.create(repository['name'], repository)
        self.jobs = []

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def add_job(self, name, config):
        """
        Add computational job to the workflow.

        Args:
            job:

        Returns:

        """
        run_ids = [j.run_id for j in self.jobs]
        if config['run_id'] in run_ids:
            self.log.warning(f'Job {name} found with run_id: {config["run_id"]}. Skipping.')
            return None
        try:
            job = job_builder.create(name, config)
        except ValueError:
            raise CSEPSchedulerException('Unable to build forecast job object.')
        self.jobs.append(job)
        return job

    def prepare(self, archive=False, dry_run=False):
        """
        Creates computing environments for each job in workflow.

        Returns:
            None

        Throws:
            OSError
        """
        for job in self.jobs:
            job.prepare(dry_run=dry_run)

        if archive:
            self.archive()

    def add_system(self, name):
        """
        Resources corresponding to keys in the system configuration dictionary.


        Args:
            name (str): identifier for the system represented with the 'name' key in the config dict

        Returns:
            None
        """
        try:
            config = machine_config[name]
        except Exception as e:
            self.log.exception(e)
            raise
        self.log.info(f"Created {name} system to use as default system.")
        self.system = system_builder.create(config['type'], config)
        self.default_system = name
        return

    def add_repository(self, config):
        """
        Default repository for all Files in the workflow.

        Args:
            name (str): name corresponding to entry in repository configuration dict.

        Returns:

        """
        try:
            name = config['name']
            repository = repo_builder.create(name, config)
            self.repository = repository
        except Exception as e:
            self.log.exception(e)
            raise
        return

    def to_dict(self):
        exclude = ['jobs','_log']
        out = {}
        for k, v in self.__dict__.items():
            if not callable(v) and k not in exclude:
                if hasattr(v, 'to_dict'):
                    new_v = v.to_dict()
                else:
                    new_v = v
                out[k] = new_v
        # custom implementation for jobs
        out["jobs"] = {}
        for job in self.jobs:
            out["jobs"][job.run_id]=job.to_dict()
        return out

    @classmethod
    def from_dict(cls, adict):
        exclude = ['jobs','repository','_log']
        try:
            repo = adict['repository']
        except KeyError:
            repo = None
        out = cls(repository=repo)
        for k,v in out.__dict__.items():
            if k not in exclude:
                try:
                    if hasattr(v, 'from_dict'):
                        new_v = v.from_dict(adict[k])
                    else:
                        new_v = adict[k]
                    setattr(out, k, new_v)
                except KeyError:
                    # use default values from constructor
                    pass

        # right now adding jobs through the setter method, not constructor
        for key in adict['jobs']:
            job_state = adict['jobs'][key]
            name = job_state['name']
            out.add_job(name, job_state)
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
        if self.repository is None:
            self.log.warning("Unable to archive simulation manifest. Repository must not be None.")
            return
        if not self.repository:
            self.log.info("Unable to access repository. Defaulting to FileSystem repository and storing in the experiment directory.")
            repo = repo_builder.create("filesystem", url=self.work_dir)
        else:
            repo = self.repository
            self.log.info(f"Found repository. Using {repo.name} to store class state.")

        # need to set here, bc archive is not explicitly called from jobs.
        for job in self.jobs:
            job.last_modified_datetime = datetime.datetime.now()

        # access storage through the repository layer
        # for sqlalchemy, this would create the Base objects to insert into the database.
        repo.save(self.to_dict(), backup=True)

    @classmethod
    def load(cls, repo):
        """
        Returns new class object using the repository stored with the class. Maybe this should be a class
        method.

        """
        if isinstance(repo, Repository):
            out=repo.load(cls())
        elif type(repo) == dict:
            repo = repo_builder.create(repo['name'], repo)
            out=repo.load(cls())
        else:
            raise CSEPSchedulerException("Unable to load state. Repository must not be None.")
        return out

    def run(self, force=True):
        # for now jobs should be responsible for running themselves
        # this represents a convenience wrapper. the manager should be used for
        # monitoring jobs.
        for job in self.jobs:
            status = job.status
            if status == 'submitted' or status == 'complete':
                if not force:
                    self.log.info(f'{job.run_id} complete or submitted. skipping.')
                else:
                    self.log.info(f"forcing execution of {job.run_id}. data will be overwritten.")
                    job.run()
            else:
                job.run()
        # archive after all jobs have run.
        self.archive()


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
        self.log.info(f'Adding {name} forecast to experiment.')
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
