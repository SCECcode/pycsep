import os
import time
from csep.core.analysis import run_u3etas_calculation
from csep.core.environment import generate_local_airflow_environment_test
from csep.core.docker import build_run_image

"""
author: william wsavran
date: 09.25.2018

general script to execute u3etas benchmark stored in docker images using csep2 modules arranged
as a script
"""

# will contain parameters for the run
# (num_sim, num_thread, memory)

# associated with benchmark part 1
# params = [(50,10,50), (100,10, 50), (200,10,50), (500, 20, 100)]

# associated with benchmark part 3 -- failed due to memory constraints
# params = [(200,10,50), (500, 20, 100)]

# benchmark part 4 -- debugging run
# params = [(20,20,100)]

# benchmark part 5 -- debugging run
# params = [(20,20,100)]

# benchmark part 6 -- debugging run
# params = [(10,10,50), (5,5,25), (1,1,5)]

# benchmark part 7
# params = [(500,20,100)]

# benchmark part 8
params = [(500,20,100)]

for num_sim, num_thread, memory in params:

    # generate workflow environment
    config = generate_local_airflow_environment_test(experiment_name='u3etas-benchmark',
                                                     model_dir='ucerf3-etas',
                                                     config_filename='u3etas_inputs.json')

    # build docker image
    config = build_run_image(config, updated_inputs={'numSimulations': num_sim,
                                                     'duration': 0.0833333333})
    # run calculation
    t0 = time.time()
    run_u3etas_calculation(config,
            environment = {'ETAS_MEM_GB': memory,
                'ETAS_LAUNCHER': '/run_dir',
                'ETAS_OUTPUT': '/run_dir/user_output',
                'ETAS_THREADS': num_thread},
            command = ["u3etas_launcher.sh", os.path.join('/run_dir', config['config_filename'])])
    t1 = time.time()
    print('time for executing run_u3etas_calculation: {}'.format(t1-t0))

    # command for post-processing if we want to run
    # command = ["u3etas_plot_generator.sh", os.path.join('/run_dir', config['config_filename']),
    #     "/run_dir/user_output/results_complete.bin"],
