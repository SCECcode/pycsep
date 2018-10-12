import os
from csep.core.analysis import run_u3etas_calculation


# setup runtime configuration
config = {'runtime_dir': os.path.expanduser('~/simulations/u3etas-benchmark/runs/u3etas-benchmark__2018-10-03T17-52-31'),
            'image_tag': 'wsavran/csep:u3etas-benchmark__2018-10-03T17-52-31'} 

# run post-processing
command = ["u3etas_plot_generator.sh", "/run_dir/u3etas_inputs.json", "/run_dir/user_output/results_complete.bin"]
environment = {'ETAS_MEM_GB': 100, 'ETAS_LAUNCHER': '/run_dir', 'ETAS_OUTPUT': '/run_dir/user_output', 'ETAS_THREADS': 20}
run_u3etas_calculation(config, environment=environment, command=command)
