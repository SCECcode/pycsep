from unittest import TestCase
from csep.core.jobs import UCERF3Forecast


class TestUCERF3Forecast(TestCase):


    def test_create(self):
        ucerf3test_config = {
            'name': 'ucerf3-etas',
            'run_id': 'testing',
            'system': 'csep-cert',
            'command': '',
            'args': '',
            'inputs': [
                '$ETAS_LAUNCHER/inputs/u3_historical_catalog.txt',
                '$ETAS_LAUNCHER/inputs/u3_historical_catalog_finite_fault_mappings.xml',
                '$ETAS_LAUNCHER/inputs/cache_fm3p1_ba',
                '$ETAS_LAUNCHER/inputs/2013_05_10-ucerf3p3-production-10runs_COMPOUND_SOL_FM3_1_SpatSeisU3_MEAN_BRANCH_AVG_SOL.zip'
            ],
            'work_dir': 'test',
            'model_dir': 'test',
            'config_templ': 'test',
            'script_templ': 'test',
            'output_dir': 'test',
            'nnodes': 18,
            'max_run_time': '12:00:00'
        }
        test_obj = UCERF3Forecast(**ucerf3test_config)
        assert test_obj.name == ucerf3test_config['name']
        assert test_obj.run_id == ucerf3test_config['run_id']
        assert test_obj._system.name == ucerf3test_config['system']
        assert test_obj.command == ucerf3test_config['command']
        assert test_obj.args == ucerf3test_config['args']
        assert test_obj._inputs == ucerf3test_config['inputs']
        assert test_obj.work_dir == ucerf3test_config['work_dir']
        assert test_obj.model_dir == ucerf3test_config['model_dir']
        assert test_obj._config_templ == ucerf3test_config['config_templ']
        assert test_obj._script_templ == ucerf3test_config['script_templ']
        assert test_obj.output_dir == ucerf3test_config['output_dir']
        assert test_obj.nnodes == ucerf3test_config['nnodes']
        assert test_obj.max_run_time == ucerf3test_config['max_run_time']