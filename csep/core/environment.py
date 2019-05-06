import os
import sys
import string
import datetime
import logging


def generate_local_airflow_environment_test(*args, **kwargs):
    """
    Configures run-time environment for CSEP experiment. Configuration file will get populated and stored inside the runtime directory.
    The file contains:
        - run_id: ID representing execution of DAG
        - log_file: filepath to logfile for DAG run
        - experiment_directory: top-level directory for experiment
        - runtime_dir: temporary directory containing the runtime contents of DAG. contains config files, etc.
        - archive_directory: directory containing archived data products from DAG execution.

    Note: This implementation is Airflow specific. We should have some functionality (maybe decorators) to indicate which workflow tool will be used
    by the different functions. In general, the base functionality should be able to run on its own.

    :param **kwargs: contains context passed to function my airflow
    type **kwargs: dict
    """
    config = {}
    csep_home = os.environ['CSEP_DEV']
    csep_models = os.environ['CSEP_MODELS']

    # parse from airflow context
    runtime = kwargs.pop('ts', datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    runtime = runtime.replace(":", "-")
    experiment_name = kwargs.pop('experiment_name', 'unknown')
    experiment_dir = kwargs.pop('experiment_dir', os.path.join(csep_home, experiment_name))
    model_dir = kwargs.pop('model_dir', None)
    if model_dir is not None:
        model_dir = os.path.join(csep_models, model_dir)
    else:
        print('Error: CSEP_MODELS environment variable must be set. Exiting.')
        sys.exit(-1)

    config_filename = kwargs.pop('config_filename')

    run_id = kwargs.pop('run_id', '__'.join([experiment_name, runtime]))

    logging.info('Configuring local run-time environment for run_id: {}.'.format(run_id))

    # generate filepath for unique runtime
    run_dir = os.path.join(experiment_dir, 'runs', run_id)

    # updates to configuration file
    config['run_id'] = run_id
    config['experiment_name'] = experiment_name
    config['experiment_dir'] = experiment_dir
    config['execution_runtime'] = runtime
    config['runtime_dir'] = run_dir
    config['model_dir'] = model_dir
    config['config_filename'] = config_filename

    # write configuration to logging
    logging.info(config)

    # make necessary directories
    os.makedirs(config['experiment_dir'], exist_ok=True)

    # runtime directory should be unique
    os.makedirs(config['runtime_dir'])

    # get values for config mapping
    fp = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(fp, '../artifacts/runtime_config.tmpl'), 'r') as template_file:
        template = string.Template(template_file.read())

    with open(os.path.join(config['runtime_dir'], 'run_config.txt'), 'w') as config_file:
        config_file.writelines(template.substitute(config))

    # push information back to airflow scheduler by returning
    return config


class SimulationConfiguration:

    def __init__(self, run_id, experiment_name, experiment_dir, machine_name,
                 execution_runtime, runtime_dir, job_script, model_dir, output_dir, config_filename,
                 template_dir, input_files=()):

        # primary index for simulation components
        self.run_id = run_id

        # human-readable name for experiment
        self.experiment_name = experiment_name

        # top-level directory holding all forecasts for that experiment
        self.experiment_dir = experiment_dir

        # name of the machine used for computations
        self.machine_name = machine_name

        # datetime when submitted to host machine
        self.execution_runtime = execution_runtime

        # directory where simulation is ran on machine name
        self.runtime_dir = runtime_dir

        # absolute path to the job run script
        self.job_script = job_script

        # directory storing the code for the model
        self.model_dir = model_dir

        # absolute path to the job configuration script
        self.config_filename = config_filename

        # directory storing template files (e.g., machine specific runtime scripts)
        self.template_dir = template_dir

        # runtime output directory
        self.output_dir = output_dir

        # iterable of input_files, this is used for archiving purposes.
        self.input_files = input_files

    @classmethod
    def from_dict(cls, adict):
        """
        Creates class from dict of objects. Useful for creating this model from repository layer.

        Args:
            adict: dictionary of parameters

        Returns:
            sim (SimulationConfiguration): object of type SimulationConfiguration

        """
        return cls(
            run_id=adict['run_id'],
            experiment_name=adict['experiment_name'],
            experiment_dir=adict['experiment_dir'],
            machine_name=adict['machine_name'],
            execution_runtime=adict['execution_runtime'],
            runtime_dir=adict['runtime_dir'],
            job_script=adict['job_script'],
            model_dir=adict['model_dir'],
            config_filename=adict['config_filename'],
            template_dir=adict['template_dir'],
            output_dir=adict['output_dir']
        )

    def to_dict(self):
        """
        Create dict of class arguments for serialization, preferably using json.

        Returns:
            dict: representation of class state, serializable.

        """
        return {
            'run_id': str(self.run_id),
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'machine_name': self.machine_name,
            'execution_runtime': self.execution_runtime,
            'runtime_dir': self.runtime_dir,
            'job_script': self.job_script,
            'model_dir': self.model_dir,
            'output_dir': self.output_dir,
            'config_filename': self.config_filename,
            'template_dir': self.template_dir
        }

    def _create_runtime_directory(self, force=True):
        """
        Creates directory for the simulation.

        Args:
            force:

        Returns:

        Throws:
            FileNotFoundError

        """
        raise NotImplementedError

    def _copy_input_files(self):
        """
        Copies input files to runtime directory.

        Returns:

        Throws:
            FileNotFoundError
        """
        raise NotImplementedError

    def _write_configuration(self, updated_inputs=None):
        """
        writes input configuration file for individual simulation. assumes that model inputs are json serializable.

        Args:
             updated_inputs: dict of inputs to overwrite

        Returns:
            None

        Throws:
            IOError: issue reading/writing file
            TypeError: issue with types contained in dictionary
        """

    def prepare(self):
        """
        Prepares simulation for submission on the compute host. This function does not submit the job on the host.

        Returns:
            None

        Throws:
            

        """
        raise NotImplementedError

