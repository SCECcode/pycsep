from unittest import TestCase
from csep.core.environment import SimulationConfiguration
import uuid


class TestSimulation(TestCase):
    def test_create_runtime_directory(self):
        run_id = uuid.uuid4()
        experiment_name = 'test-experiment'
        experiment_dir = '/home/testuser/test_dir'
        machine_name = 'csep-op'
        execution_runtime = 'my-datetime'
        runtime_dir = '/home/testuser/test_runtime_dir'
        job_script = 'my_run_script.csh'
        model_dir = 'model_dir'
        config_filename = 'my_config_filename.json'
        output_dir = 'test_output_dir'
        template_dir = '/home/testuser/path_to_template'

        sim = SimulationConfiguration(run_id, experiment_name, experiment_dir, machine_name,
                                      execution_runtime, runtime_dir, job_script, model_dir, output_dir,
                                      config_filename, template_dir)

        self.assertEqual(sim.run_id, run_id)
        self.assertEqual(sim.experiment_dir, experiment_dir)
        self.assertEqual(sim.experiment_name, experiment_name)
        self.assertEqual(sim.machine_name, machine_name)
        self.assertEqual(sim.execution_runtime, execution_runtime)
        self.assertEqual(sim.runtime_dir, runtime_dir)
        self.assertEqual(sim.job_script, job_script)
        self.assertEqual(sim.model_dir, model_dir)
        self.assertEqual(sim.output_dir, output_dir)
        self.assertEqual(sim.config_filename, config_filename)
        self.assertEqual(sim.template_dir, template_dir)

    def test_create_simulation_from_dict(self):
        adict = {
            'run_id': uuid.uuid4(),
            'experiment_name': 'test-experiment',
            'experiment_dir': '/home/testuser/test_dir',
            'machine_name': 'csep-op',
            'execution_runtime': 'my-datetime',
            'runtime_dir': '/home/testuser/test_runtime_dir',
            'job_script': 'my_run_script.csh',
            'model_dir': 'model_dir',
            'output_dir': 'output_dir',
            'config_filename': 'my_config_filename.json',
            'template_dir': '/home/testuser/path_to_template'
        }
        sim = SimulationConfiguration.from_dict(adict)

        self.assertEqual(sim.run_id, adict['run_id'])
        self.assertEqual(sim.experiment_dir, adict['experiment_dir'])
        self.assertEqual(sim.experiment_name, adict['experiment_name'])
        self.assertEqual(sim.machine_name, adict['machine_name'])
        self.assertEqual(sim.execution_runtime, adict['execution_runtime'])
        self.assertEqual(sim.runtime_dir, adict['runtime_dir'])
        self.assertEqual(sim.job_script, adict['job_script'])
        self.assertEqual(sim.model_dir, adict['model_dir'])
        self.assertEqual(sim.output_dir, adict['output_dir'])
        self.assertEqual(sim.config_filename, adict['config_filename'])
        self.assertEqual(sim.template_dir, adict['template_dir'])

    def test_to_dict_equal_from_dict(self):
        adict = {
            'run_id': uuid.uuid4(),
            'experiment_name': 'test-experiment',
            'experiment_dir': '/home/testuser/test_dir',
            'machine_name': 'csep-op',
            'execution_runtime': 'my-datetime',
            'runtime_dir': '/home/testuser/test_runtime_dir',
            'job_script': 'my_run_script.csh',
            'model_dir': 'model_dir',
            'output_dir': 'output_dir',
            'config_filename': 'my_config_filename.json',
            'template_dir': '/home/testuser/path_to_template'
        }
        sim = SimulationConfiguration.from_dict(adict)

        self.assertEqual(sim.to_dict(), SimulationConfiguration.from_dict(adict).to_dict())
