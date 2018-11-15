import os
import sys
import uuid
import string
import datetime
import logging
import shutil


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
