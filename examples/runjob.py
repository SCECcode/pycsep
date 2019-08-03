import os
from csep.core.jobs import BaseTask

# this example will print your home directory

run_id = 'find job'
system = 'default'
run_command = 'ls'
command_args = f"{os.path.expanduser('~/Desktop')}"
work_dir = None
inputs = ()
outputs = ()

job = BaseTask(run_id=run_id, system=system, command=run_command, args=command_args,
               work_dir=work_dir, inputs=inputs, outputs=outputs)
job.run()

