import os
import docker
from csep.core.environment import generate_local_airflow_environment_test

"""
author: william wsavran
date: 09.25.2018

general script to execute u3etas stored in docker images using csep2 modules arranged
as a script
"""


def run_u3etas_calculation(**kwargs):
    """
    run u3etas with new user interface.

    :param **kwargs: contains the context provided by airflow
    type: dict
    """
    # get configuration dict from scheduler
    ti = kwargs.pop('ti')
    config = ti.xcom_pull(task_ids='generate_environment')

    # setup docker using easy interfact
    host_dir = os.path.join(config['runtime_dir'], 'output_dir')
    container_dir = '/run_dir/user_output'

    client = docker.from_env()
    container = client.containers.run('wsavran/csep:u3etas-test2',
            volumes = {host_dir:
                {'bind': container_dir, 'mode': 'rw'}},
            environment = {'ETAS_MEM_GB': '14',
                'ETAS_LAUNCHER': '/run_dir',
                'ETAS_OUTPUT': '/run_dir/user_output',
                'ETAS_THREADS': '3'},
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line.decode('utf-8'))


def run_u3etas_post_processing(**kwargs):
    """
    run post-processing for u3etas

    :param **kwargs: context passed from airflow scheduler
    type: dict
    """
    # get configuration dict from scheduler
    ti = kwargs.pop('ti')
    config = ti.xcom_pull(task_ids='generate_environment')

    # setup docker using easy interfact
    host_dir = os.path.join(config['runtime_dir'], 'output_dir')
    container_dir = '/run_dir/user_output'

    client = docker.from_env()
    container = client.containers.run('wsavran/csep:u3etas-test2',
            volumes = {host_dir:
                {'bind': container_dir, 'mode': 'rw'}},
            environment = {'ETAS_MEM_GB': '14',
                'ETAS_LAUNCHER': '/run_dir',
                'ETAS_OUTPUT': '/run_dir/user_output',
                'ETAS_THREADS': '3'},
            command = ["u3etas_plot_generator.sh", "/run_dir/config.json", "/run_dir/user_output/results_complete.bin"],
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line)

config = generate_local_airflow_environment_test(experiment_name='u3etas-benchmark')

run_u3etas_calculation(config=config)
