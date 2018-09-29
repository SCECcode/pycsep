import os
import sys
import shutil
import docker
import json

def build_run_image(config, updated_inputs=None):
    """
    builds a docker image that will be used for experiments. this function needs
    the global experiment configuration and a dictionary of inputs that need to be
    updated from the generic configuration parameters
    """

    # IDEA: add this functionality into the setup environment step
    # grab useful information from config
    model_dir = config['model_dir']
    base_config_file = os.path.join(model_dir, config['config_filename'])
    runtime_dir = config['runtime_dir']
    run_config_file = os.path.join(runtime_dir, config['config_filename'])
    run_dockerfile = os.path.join(model_dir, 'dockerfile')
    run_id = config['run_id']

    # copy input file from model directory to runtime directory
    shutil.copy(base_config_file, runtime_dir)
    shutil.copy(run_dockerfile, runtime_dir)

    # hardcoding configuration for json input files, but we will need
    # support for multiple types of configuration files
    with open(run_config_file, 'r') as f:
        model_config = json.load(f)
    model_config.update(updated_inputs)
    with open(run_config_file, 'w') as f:
        json.dump(model_config, f)

    # build docker image with updated runtime file
    container_tag = 'wsavran/csep:' + run_id
    cli = docker.APIClient(base_url='unix://var/run/docker.sock')
    for line in cli.build(path=runtime_dir, tag=container_tag, rm=True):
        print(line.decode('utf-8'))

    # update config with new image tag
    config['container_tag'] = container_tag

    # updated config
    return config
    
