import os
import json
from csep.core.managers import ForecastExperiment
from csep.utils.constants import SECONDS_PER_WEEK

experiment_config = {
    "name": "UCERF3-ETAS Aftershock Study",
    "description": "Example study using UCERF3-ETAS and UCERF3-NoFaults immediately following major earthquakes in California.",
    "system": "hpc-usc",
    "base_dir": '/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/runs',
    "responsible_scientist": ["bill", "max"]
}

ucerf3job_config = {
    'name': 'ucerf3-etas',
    'system': 'hpc-usc',
    'command': None,
    'args': None,
    'inputs': [
        '$ETAS_LAUNCHER/inputs/u3_historical_catalog.txt',
        '$ETAS_LAUNCHER/inputs/u3_historical_catalog_finite_fault_mappings.xml',
        '$ETAS_LAUNCHER/inputs/cache_fm3p1_ba',
        '$ETAS_LAUNCHER/inputs/2013_05_10-ucerf3p3-production-10runs_COMPOUND_SOL_FM3_1_SpatSeisU3_MEAN_BRANCH_AVG_SOL.zip'
    ],
    'model_dir': '/Users/wsavran/Projects/Code/ucerf3-etas-launcher',
    'config_templ': '/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/template_files/ucerf3-defaults.json',
    'script_templ': '/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/template_files/hpc-usc.slurm',
    'output_dir': '/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/runs',
    'nnodes': 18,
    'max_run_time': '12:00:00'
}

# start time in milisecons for Landers rupture
epoch_time = 709732655000

# generate start times every 1 week
origin_times = [epoch_time + i * SECONDS_PER_WEEK for i in range(4)]

# start with experiment shell
# serializable so we can load experiment back into memory (or on database)
exp1 = ForecastExperiment(experiment_config)

# iterate through variable parameters in experiment
for origin_time in origin_times:
    # add UCERF3-ETAS Forecast
    run_id = f'ot_{origin_time}'
    ucerf3job_config.update({'work_dir': f'/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/runs/{run_id}',
                             'run_id': run_id})
    fore=exp1.add_forecast(ucerf3job_config, force=True)
    fore.update_configuration({"startTimeMillis": origin_time,
                               "duration": 1.0,
                               "numSimulations": 10000})
    fore.add_output("results_complete.bin", "binary")

    # add UCERF3-ETAS NoFaults Forecast
    run_id = f'ot_{origin_time}_nf'
    ucerf3job_config.update({'work_dir': f'/Users/wsavran/Projects/Code/ucerf3_run_gen_testing/runs/{run_id}',
                             'run_id': run_id})
    fore=exp1.add_forecast(ucerf3job_config, force=True)
    fore.update_configuration({"startTimeMillis": origin_time,
                               "duration": 1.0,
                               "numSimulations": 10000,
                               "griddedOnly": True,
                               "totRateScaleFactor": 1.0,
                               "gridSeisCorr": False})
    fore.add_output("results_complete.bin", "binary")

# manage all computing environments needed to run the job
exp1.prepare()
mfile = os.path.join(exp1.base_dir, 'trial_u3etas_aftershock_manifest.json')
with open(mfile, 'w') as f:
    print(f'Writing archive file to {mfile}.')
    json.dump(exp1.to_dict(), f, indent=4, separators=(',', ': '))

# submit all jobs to compute nodes
exp1.run()

# write new state.
mfile = os.path.join(exp1.base_dir, 'trial_u3etas_aftershock_manifest.json')
with open(mfile, 'w') as f:
    print(f'Writing archive file to {mfile}.')
    json.dump(exp1.to_dict(), f, indent=4, separators=(',', ': '))

