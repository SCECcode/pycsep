import os
import subprocess
import string
import json
from collections import namedtuple

from csep.core.factories import ObjectFactory
from csep.utils.log import LoggingMixin

SlurmJobStatus = namedtuple('JobStatus', ['returncode', 'status', 'job_id', 'type', 'stdout', 'stderr'])

class System(LoggingMixin):
    def __init__(self, name=None, url=None, hostname=None,
                 email=None, max_cores=None, mem_per_node=None, **kwargs):
        self.name = name
        self.url = url
        self.hostname = hostname
        self.email = email
        self.max_cores = max_cores
        self.mem_per_node = mem_per_node

    def __eq__(self, other):
        try:
            return self.to_dict() == other.to_dict()
        except:
            return False

    def __str__(self):
        return self.to_dict()

    @classmethod
    def from_dict(cls, adict):
        return cls(**adict)

    def to_dict(self):
        exclude = ['_log']
        out = {}
        for k, v in self.__dict__.items():
            if not callable(v) and k not in exclude:
                if hasattr(v, 'to_dict'):
                    out[k] = v.to_dict()
                else:
                    out[k] = v
        return out

    def check_environment(self, env):
        """
        Checks that environment variables of system are set and consistent with configuration
        file.

        Args:
            env (dict): dict in format {"VAR_NAME": "VAR_VALUE"}

        Returns:
            result (bool): true if correct, false if not.

        """
        res=True
        for key, val in env.items():
            test_val=os.environ.get(key)
            if test_val is None or test_val != val:
                res=False
        return res

    def execute(self, cmnd=None, args=None):
        """
        Executes a process on the system

        Args:
            cmnd (str): shell command to run
            args (str): arguments to give to command

        Returns:
            process (subprocess.CompletedProcess):
        """
        command = [cmnd, *args.split(' ')]
        out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # capture return code
        stderr = out.stderr.decode("utf-8")
        rc = out.returncode
        if rc != 0:
            self.log.error(f"Could not executing command {' '.join(command)}.\n{stderr}")
        else:
            print(out.stdout.decode('utf-8'))
        return out


class SlurmSystem(System):
    def __init__(self, *args, **kwargs):
        # get args here first then pass to parent
        self.mpj_home = kwargs.pop('mpj_home')
        self.partition = kwargs.pop('partition')
        super().__init__(*args, **kwargs)

    def execute(self, cmnd='sbatch', args=(), run_dir=None):
        """
        Execute a batch job on a Slurm system. This job adds some addition output than the base machine.
        The job_id can be use by the monitor to determine the status of various jobs.

        Example:
            $$$ sbatch my_slurm_job.slurm

            slurm.execute(args='my_slurm_job.slurm', cmnd='sbatch')

        Args:
            args (List(str): command args, ie the name of the script'
            cmnd (str): program to run ie, 'sbatch'

        Returns:
            (dict) with additional slurm parameters
        """
        cmnd = 'sbatch'
        # do not need to be in the directory to launch the script
        if run_dir is not None:
            dir_args = ['-D', run_dir]
        else:
            dir_args = []
        # cmnd is a single string command (e.g., 'sbatch'), args should be an iterable
        full_cmnd = [cmnd] + dir_args + args
        self.log.info(f"Executing {' '.join(full_cmnd)}.")
        out = subprocess.run(full_cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # capture return code
        stdout = out.stdout.decode("utf-8")
        stderr = out.stderr.decode("utf-8")
        if out.returncode == 0:
            self.log.info(f'Successfully submitted job to slurm Scheduler with job_id: {stdout.split()[-1]}')
            status = SlurmJobStatus(returncode=out.returncode,
                                    status='Success',
                                    job_id=stdout.split()[-1],
                                    type='batch',
                                    stdout=stdout,
                                    stderr=stderr)
        else:
            status = SlurmJobStatus(returncode=out.returncode,
                                    status='Failed',
                                    job_id=None,
                                    type='batch',
                                    stdout=None,
                                    stderr=None)
            self.log.error(f"Error with batch submission.\n{stderr}")
        return status


class File(LoggingMixin):
    def __init__(self, path):
        self.type='base'
        self.path = os.path.expanduser(os.path.expandvars(str(path)))
        self._handle = None

    def __del__(self):
        if self._handle:
            self._handle.close()

    def to_dict(self):
        return {'path': self.path,
                'type': self.type}

    @classmethod
    def from_dict(cls, adict):
        return cls(**adict)

    def template(self, data):
        raise NotImplementedError

    def open(self):
        raise NotImplementedError

    def close(self):
        self._handle.close()

    def write(self, path):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


class TextFile(File):
    def __init__(self, path):
        super().__init__(path)
        self.type = 'text'
        self._contents = None

    def open(self, mode='r'):
        self.log.info(f"Opening file at {self.path}.")
        self._handle = open(self.path, mode)

    def template(self, adict):
        if self._contents is None:
            self.read()
        template = string.Template(self._contents)
        new_contents=template.substitute(adict)
        self._contents = new_contents
        return self

    def read(self):
        if self._handle is None:
            self.open()
        self._contents = self._handle.read()
        self.close()

    def write(self, new_path=None):
        if new_path:
            self.path=new_path
        self.open(mode='w')
        self._handle.writelines(self._contents)
        self.close()

class BinaryFile(File):
    def __init__(self, path):
        super().__init__(path)
        self.type='binary'

    def open(self, mode='rb'):
        self._handle = open(self.path, mode)

class JsonFile(TextFile):
    def __init__(self, path):
        super().__init__(path)
        self.type='json'

    def template(self, adict):
        if self._contents is None:
            self.read()
        self._contents.update(adict)
        self.close()
        return self

    def read(self):
        if self._handle is None:
            self.open()
        self._contents = json.load(self._handle)
        self.close()

    def write(self, new_path=None):
        if new_path:
            self.path=new_path
        self.open(mode='w')
        json.dump(self._contents, self._handle,
                  indent=4, separators=(',', ': '))
        self.close()


# Register System builders
system_builder = ObjectFactory()
system_builder.register_builder('default', System.from_dict)
system_builder.register_builder('slurm', SlurmSystem.from_dict)
system_builder.register_builder('hpc-usc', SlurmSystem.from_dict)

# File Builder Objects
file_builder = ObjectFactory()
file_builder.register_builder('binary', BinaryFile)
file_builder.register_builder('text', TextFile)
file_builder.register_builder('json', JsonFile)