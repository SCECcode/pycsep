import logging
import os
import tempfile

import unittest
from csep.core.managers import ForecastExperiment
from csep.utils.constants import SECONDS_PER_WEEK

class AcceptanceTests(unittest.TestCase):

    def test_create_and_load_experiment(self):

        tmp_dir = tempfile.TemporaryDirectory()
        experiment_config = {
            "name": "Test Study",
            "description": "Tests are good ",
            "base_dir": '',
            "owner": ["csep tester"]
        }

        repository_config = {
            "name": "filesystem",
            "url": os.path.join(tmp_dir.name, "test-manifest.json")
        }

        machine_config = {
            "hpc-usc": {
                "name": "hpc-usc",
                "url": "hpc.usc.edu",
                "hostname": "hpc-login",
                "email": "noemail@scec.org",
                "mpj_home": "test",
                "partition": "scec",
                "max_cores": 20,
                "mem_per_node": 64
            },
            "default": {
                "name": "default",
                "url": "default",
                "hostname": "default",
                "email": "None",
                "type": "direct",
                "max_cores": 1,
                "mem_per_node": 2,
            }
        }

        root_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root_dir, '../csep/artifacts/Configurations')
        ucerf3job_config = {
            'name': 'ucerf3-etas',
            'system': machine_config['hpc-usc'],
            'command': 'sbatch',
            'args': None,
            'inputs': [],
            'model_dir': 'test',
            'config_templ': os.path.join(data_dir, 'ucerf3-defaults.json'),
            'script_templ': os.path.join(data_dir, 'hpc-usc.slurm'),
            'output_dir': os.path.join(tmp_dir.name, 'runs'),
            'nnodes': 18,
            'max_run_time': '12:00:00'
        }

        forecast_config = {
            "ucerf3-etas":
                {
                    "config_templ": os.path.join(data_dir, 'ucerf3-defaults.json'),
                    "script_templ": os.path.join(data_dir, 'hpc-usc.slurm'),
                    "env": {"ETAS_LAUNCHER": "testing"}
                }
        }

        # start time in milisecons for Landers rupture
        epoch_time = 709732655000

        # generate start times every 1 week
        origin_times = [epoch_time + i * SECONDS_PER_WEEK for i in range(10)]

        # start with experiment shell
        # serializable so we can load experiment back into memory (or on database)
        exp1 = ForecastExperiment(**experiment_config)

        # add repository to save experiment information
        exp1.add_repository(repository_config)

        # iterate through variable parameters in experiment
        for origin_time in origin_times:
            # add UCERF3-ETAS Forecast
            run_id = f'ot_{origin_time}'
            ucerf3job_config.update({'work_dir': f'/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/runs/{run_id}',
                                     'run_id': run_id})
            fore=exp1.add_forecast(ucerf3job_config)
            fore.update_configuration({"startTimeMillis": origin_time,
                                       "duration": 1.0,
                                       "numSimulations": 10000})
            fore.add_output("results_complete.bin")

        # manage all computing environments needed to run the job
        exp1.prepare(archive=True, dry_run=True)
        exp2 = ForecastExperiment.load(repository_config)

        # # load the experiment in from the command-line
        assert exp1 == exp2
        exp2.archive()

        exp3 = ForecastExperiment.load(repository_config)
        assert exp2 == exp3