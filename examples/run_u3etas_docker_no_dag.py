import os
from csep.core.analysis import run_u3etas_calculation
from csep.core.environment import generate_local_airflow_environment_test

"""
author: william wsavran
date: 09.25.2018

general script to execute u3etas stored in docker images using csep2 modules arranged
as a script
"""

config = generate_local_airflow_environment_test(experiment_name='u3etas-benchmark')

run_u3etas_calculation(config=config)
