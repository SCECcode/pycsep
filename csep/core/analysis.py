import docker
import os

def run_u3etas_calculation(config, *args, **kwargs):
    """
    run u3etas with new user interface.

    :param **kwargs: contains the context provided by airflow
    type: dict
    """
    # get configuration dict from scheduler
    ti = kwargs.pop('ti', None)
    if ti is not None:
        config = ti.xcom_pull(task_ids='generate_environment')

    # setup docker using easy interface
    host_dir = os.path.join(config['runtime_dir'], 'output_dir')
    container_dir = '/run_dir/user_output'

    client = docker.from_env()
    container = client.containers.run(config['container_tag'],
            volumes = {host_dir:
                {'bind': container_dir, 'mode': 'rw'}},
            environment = {'ETAS_MEM_GB': '6',
                'ETAS_LAUNCHER': '/run_dir',
                'ETAS_OUTPUT': '/run_dir/user_output',
                'ETAS_THREADS': '1'},
            command = ["u3etas_launcher.sh", os.path.join('/run_dir', config['config_filename'])],
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line.decode('utf-8'))

def run_u3etas_post_processing(config, **kwargs):
    """
    run post-processing for u3etas

    :param **kwargs: context passed from airflow scheduler
    type: dict
    """
    # get configuration dict from scheduler
    ti = kwargs.pop('ti', None)
    if ti is not None:
        config = ti.xcom_pull(task_ids='generate_environment')

    # setup docker using easy interfact
    host_dir = os.path.join(config['runtime_dir'], 'output_dir')
    container_dir = '/run_dir/user_output'

    client = docker.from_env()
    container = client.containers.run(config['container_tag'],
            volumes = {host_dir:
                {'bind': container_dir, 'mode': 'rw'}},
            environment = {'ETAS_MEM_GB': '6',
                'ETAS_LAUNCHER': '/run_dir',
                'ETAS_OUTPUT': '/run_dir/user_output',
                'ETAS_THREADS': '1'},
            # FIXME: need to make this simulation agnostic? or just create bespoke function for each model?
            command = ["u3etas_plot_generator.sh", os.path.join('/run_dir', config['config_filename']), 
                "/run_dir/user_output/results_complete.bin"],
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line.decode('utf-8'))
