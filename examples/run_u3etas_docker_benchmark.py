import os
import json
import logging
import time
from csep.core.analysis import run_u3etas_calculation
from csep.core.environment import generate_local_airflow_environment_test
from csep.core.docker import build_run_image

"""
author: william wsavran
date: 09.25.2018

general script to execute u3etas benchmakr stored in docker images using csep2 modules arranged
as a script
"""

num_sims = [10, 25, 50, 100, 500, 1000]
num_sims = [10, 25]

for num_sim in num_sims:

    # generate workflow environment
    config = generate_local_airflow_environment_test(experiment_name='u3etas-benchmark',
                                                     model_dir='ucerf3-etas',
                                                     config_filename='u3etas_inputs.json')

    # build docker image
    config = build_run_image(config, updated_inputs={'numSimulations': num_sim,
                                                     'duration': 1.0})
    # run calculation
    t0 = time.time()
    run_u3etas_calculation(config)
    t1 = time.time()
    print('time for executing run_u3etas_calculation: {}'.format(t1-t0))
