"""
Job classes represent computational units.
"""
import os
import uuid
import datetime

from csep.utils.file import mkdirs, copy_file
from csep.core.config import machine_config
from csep.core.system import system_builder, JsonFile, TextFile, System
from csep.core.exceptions import CSEPSchedulerException
from csep.core.factories import ObjectFactory
from csep.core.repositories import repo_builder, Repository
from csep.utils.log import LoggingMixin


class BaseTask(LoggingMixin):
    """
    Represents the base class for any job needed to run on the system.

    """

    def __init__(self, name='', run_id=None, system=None, max_run_time=None, repository=None,
                       command='', args=(), status='', inputs=(), outputs=(),
                       work_dir=''):

        # primary index for simulation components
        self.run_id = run_id or uuid.uuid4().hex
        self.status = status or 'unprepared'
        self.command = command
        self.args = args
        self.name = name
        self.work_dir = work_dir
        if self.work_dir is not None:
            self.work_dir = os.path.expandvars(os.path.expanduser(work_dir))
        self.max_run_time = max_run_time
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.prepared = False
        self.run_datetime = None
        self.last_modified_datetime = None

        # flag to warn user if trying to run in existing directory.
        self.force = False

        # should set all attributes in constructor
        self.repository = repository
        if not isinstance(self.repository, Repository):
            if type(repository) is dict:
                self.repository = repo_builder.create(repository['name'], repository)
            elif repository is None:
                # let repo be none, will use work_dir
                pass
            else:
                raise TypeError("Repository must be a dict, Repository, or None (default)")

        # system works differently, because for now we are storing system information in the package.
        self.system = system
        if not isinstance(self.system, System):
            # if system is none, use default
            if system is None:
                self.system = system_builder.create(machine_config['default']['name'], machine_config['default'])
            # if string, use default
            elif type(system) is str:
                self.system = system_builder.create(machine_config[system]['name'], machine_config[system])
            # if dict, create object from dict
            elif type(system) is dict:
                self.system = system_builder.create(system['name'], system)
            else:
                raise TypeError("system must be None (use default), string, or dict.")

    def __str__(self):
        return self.to_dict()

    def __eq__(self, other):
        try:
            return self.to_dict() == other.to_dict()
        except:
            return False

    def add_output(self, path):
        """
        Register outputs with the Job.

        Args:
            iterable (File): iterable of File objects

        Returns:

        """
        # only store the path of the file as state.
        if not os.path.isabs(path):
            new_path = os.path.join(self.work_dir, path)
        self.outputs.append(new_path)

    def add_input(self, path):
        """
        Registers Inputs with the Job.

        Args:
            iterable:

        Returns:

        """
        if not os.path.isabs(path):
            new_path = os.path.join(self.work_dir, path)
        self.inputs.append(new_path)

    def prepare(self):
        """
        Generates run-time environment necessary to execute the Job. Should be overwritten
        in subclasses.

        Args:
            None

        Return:
            None
        """
        self.prepared = True

    def run(self, print_stdout=True):
        """
        This function executes the job on the system specified by the scheduler.

        Call is blocking and will stream the output from stdout in real-time. We assume
        that all jobs are going to be running as command line processes.

        Returns:
            rc: Return code from running process.
        """
        out = self.system.execute(cmnd=self.command, args=self.args)
        if print_stdout and out.returncode == 0:
            self.log.info(out.stdout.decode("utf-8"))
            self.run_datetime = datetime.datetime.now()

    def archive(self):
        """
        Will store the state of an experiment to the repository defined using the default_repository.


        The repository layer is only used for storing the state of an object. For example, files generated during
        a job will not be saved using the repository layer. These files are typically written out by 3rd party
        programs.

        Returns:
            None
        """
        if not self.repository:
            self.log.warning("Unable to access repository. Defaulting to FileSystem repository and storing in the experiment directory.")
            repo = {'name': 'filesystem',
                    'url': os.path.join(self.work_dir, self.run_id + "-manifest.json")}
            self.repository = repo_builder.create("filesystem", repo)
        else:
            self.log.info(f"Found repository. Using {self.repository.name} to store class state.")

        # stored in local time-zone
        self.last_modified_datetime = datetime.datetime.now()

        # access storage through the repository layer
        self.repository.save(self.to_dict())

    def to_dict(self):
        """ Returns class state as JSON serializable dict. """
        exclude = ['_log']
        out = {}
        for k, v in self.__dict__.items():
            if not callable(v) and k not in exclude:
                if hasattr(v, 'to_dict'):
                    new_v=v.to_dict()
                else:
                    new_v=v
                out[k] = new_v
        return out

    @classmethod
    def from_dict(cls, adict):
        exclude = ['system', 'repository','_log']
        # handle special attributes
        system = adict.get('system', None)
        repo = adict.get('repository', None)
        lmd = adict.get('last_modified_datetime', None)
        if lmd:
            adict['last_modified_datetime'] = datetime.datetime.strptime(lmd, '%Y-%m-%d %H:%M:%S.%f')
        rd = adict.get('run_datetime', None)
        if rd:
            adict['run_datetime'] = datetime.datetime.strptime(rd, '%Y-%m-%d %H:%M:%S.%f')
        out = cls(system=system, repository=repo)
        for k,v in out.__dict__.items():
            if k not in exclude:
                try:
                    if hasattr(v, 'from_dict'):
                        new_v = v.from_dict(adict[k])
                    else:
                        new_v = adict[k]
                    setattr(out, k, new_v)
                except KeyError:
                    # ignore and use default values from constructor
                    pass
        return out

    def _stage_inputs(self, copy=False):
        """
        Stages Model inputs according to the user-defined plan.

        Returns:
            None

        Throws:
            FileNotFoundError
        """
        """
        Make sure that input files are in the correct locations.

        Note: Could use File Objects in the future, right now just assuming
        they are strings.

        Returns:

        """
        n_inputs = len(self.inputs)
        if copy:
            self.log.info(f"Staging {n_inputs} inputs to {self.work_dir}.")
            for inp in self.inputs:
                self.log.info(f"Copying {inp} to {self.work_dir}.")
                copy_file(inp, self.work_dir)
        else:
            self.log.info(f"Found {n_inputs} listed, but not copying. Maintaining for archival.")

    def _create_environment(self):
        """
        Creates compute environment to run the job.

        Args:
            force (bool): overwrites directory if True

        Returns:

        Throws:
            FileNotFoundError

        """
        """
        Create run-time environment needed to run UCERF3-ETAS

        Returns:

        """
        if os.path.isdir(self.work_dir):
            if self.force:
                self.log.warning(f'Warning: Found directory at {self.work_dir}. Forcing overwrite.')
            else:
                raise CSEPSchedulerException("Working directory already exists. Set force = True to overwrite.")
        try:
            mkdirs(self.work_dir, 0o0755)
        except Exception as e:
            self.log.exception(e)


class UCERF3Forecast(BaseTask):
    def __init__(self, model_dir='', config_templ='',
                       script_templ='', output_dir='', nnodes=None, force=False, run_script=None,
                       **kwargs):

        super().__init__(**kwargs)

        # should still work if user supplies None
        self.model_dir = os.path.expanduser(os.path.expandvars(model_dir))
        self.config_templ = os.path.expanduser(os.path.expandvars(config_templ))
        self.script_templ = os.path.expanduser(os.path.expandvars(script_templ))
        self.force = force

        # gets set if submitting an HPC Job
        self.job_id = None
        self.nnodes = nnodes

        # has the user called prepare() before. distinguishes for dry-run
        self.staged = False

        # these variables could be used for archiving purposes.
        # we could write the config or the run_script to a db or file
        # to rerun model as it existed.
        self.config = None
        self.run_script = run_script
        if self.run_script is not None:
            self.run_script = TextFile(run_script)
        # handle to config template file
        self.config = None

        # runtime output directory
        if output_dir:
            self.output_dir = os.path.expanduser(os.path.expandvars(output_dir))
        else:
            self.output_dir = self.work_dir

    def prepare(self, dry_run=False, force=True):
        """
        Create necessary environment for running the job.

        Returns:

        """
        self.work_dir = os.path.expanduser(os.path.expandvars(self.work_dir))
        self.log.info(f"Preparing UCERF3-ETAS forecast {self.name} in dir {self.work_dir}.")
        if self.staged and not force:
            self.log.info(f"UCERF3-ETAS forecast {self.name} in dir {self.work_dir} already staged. Skipping.")
        else:
            self._load_config_state()
            self.staged = True

        # state modifying action.
        if not dry_run and not self.prepared:
            self.log.info(f'Creating run-time environment for {self.name}. This action modifies state on the system.')
            self._create_environment()
            self.config.write(new_path=self.config.path)
            self.run_script.write(new_path=self.run_script.path)
            self._stage_inputs()
            self.prepared = True
            self.status = 'prepared'

    def run(self):
        if not self.prepared:
            self.prepare()
        # does not pass command, that is handled by the system.
        out = self.system.execute(args=[self.args], run_dir=self.work_dir)
        if out.returncode == 0:
            self.run_datetime = datetime.datetime.now()
            self.status = 'submitted'
            self.job_id = out.job_id
        else:
            self.status = 'failed'
        # update archive to reflect current state
        self.archive()
        return out.returncode

    def update_configuration(self, adict={}):
        """ Updates UCERF3 configuration file with correct inputs. """
        self.config = JsonFile(self.config_templ)

        # ucerf3 specific configuration parameter
        if not 'outputDir' in adict.keys():
            adict['outputDir'] = self.work_dir
        if not 'simulationName' in adict.keys():
            adict['simulationName'] = self.run_id
        # just bind adict to template file object for now, want to perform this lazily in case there is an error
        self.config.new_params = adict

    def _update_run_script(self, adict={}):
        """
        Updates run script with parameters needed to run the job.

        Returns:

        """
        if self.script_templ is None:
            raise CSEPSchedulerException("Cannot generate run-script without knowing the location"    
                                         " of the template.")
        # if none, use template, else use stored file
        if self.run_script is None:
            run_script = TextFile(self.script_templ)
            # template returns File Object
            self.run_script = run_script.template(adict)

    def _load_config_state(self):
        """ Loads all information necessary to prepare the workflow. But does not alter state in
        repository. """

        if self.config_templ is None:
            raise CSEPSchedulerException("Cannot create forecast without configuration.")

        if self.script_templ is None:
            raise CSEPSchedulerException("Cannot create forecast without run-script.")

        if self.system is None:
            raise CSEPSchedulerException("Cannot create forecast without system information.")

        # if user didn't specific a configuration, will run with values in config_templ
        if self.config is None:
            self.update_configuration()

        # template configuration parameters.
        # this operation is read-only, and stores templated values in mem.
        new = self.config.new_params
        self.config=self.config.template(new)
        self.config.path = os.path.join(self.work_dir, self.run_id + "-config.json")
        self.inputs.append(self.config.path)

        # same for the run-script
        runtime_config = self._system_runtime_config()
        self._update_run_script(runtime_config)
        self.run_script.path = os.path.join(self.work_dir, self.run_id + ".run")

        # command would be bash or sbatch, etc.
        # args would be the script
        self.args = self.run_script.path

    def _system_runtime_config(self):
        """
        Returns dict of configuration parameters necessary to update run-script.

        Returns:
            new (dict): new params that are needed to run UCERF3 on particular system
        """

        if self.system is None:
            raise CSEPSchedulerException("Cannot generate configuration information for system"
                                         " with no system bound.")
        out = {}
        if self.system.name == "hpc-usc":
            out['partition'] = self.system.partition
            out['email'] = self.system.email
            out['nodes'] = self.nnodes
            out['time'] = self.max_run_time
            out['config'] = self.config.path
            out['mpj_home'] = self.system.mpj_home
        return out

class Evaluation(BaseTask):
    pass

job_builder = ObjectFactory()
job_builder.register_builder('base', BaseTask.from_dict)
job_builder.register_builder('ucerf3-etas', UCERF3Forecast.from_dict)