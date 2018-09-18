import docker
import os

def run_u3etas(**kwargs):
    """
    task to run u3etas within airflow. uses xcom to pass state information between runs

    :param **kwargs: context passed to function from airflow
    type: dict
    """
    # get configuration dict from scheduler
    ti = kwargs['ti']
    config = ti.xcom_pull(task_ids='generate_environment')
    
    # setup docker environment
    client = docker.from_env()
    container = client.containers.run('wsavran/csep:u3etas-test', 
            volumes=['{}:/run_dir/output_dir'.format(os.path.join(config['runtime_dir'],'output_dir'))],
            detach = True,
            stderr = True)

    # stream container output to stdout
    for line in container.logs(stream=True):
        print(line)

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
        print(line)

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
            command = ["u3etas_plot_generator.sh", "/run_dir/input_catalog_with_spontaneous_example.json", "/run_dir/user_output/results_complete.bin"],
            detach = True,
            stderr = True)

    # stream output to stdout
    for line in container.logs(stream=True):
        print(line)

