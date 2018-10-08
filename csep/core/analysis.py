import docker
import os

def run_u3etas_calculation(config, environment=None, command=None, *args, **kwargs):
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
    container = client.containers.run(config['image_tag'],
            volumes = {host_dir:
                {'bind': container_dir, 'mode': 'rw'}},
            environment = environment,
            command = command,
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line.decode('utf-8'))
