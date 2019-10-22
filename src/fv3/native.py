import os
from subprocess import check_call

FV3_EXE = os.getenv('FV3_EXE', '/FV3/fv3.exe')

submit_job_script =  """
#!/bin/bash

ulimit -s unlimited
cd {rundir}
mpirun -np 6 --allow-run-as-root  \
        --mca btl_vader_single_copy_mechanism none \
        --oversubscribe /FV3/fv3.exe

"""

def write_submit_job_script(directory):
    submit_job_path = os.path.join(directory, 'submit_job.sh')
    with open(submit_job_path, "w") as f:
        f.write(submit_job_script.format(rundir=directory))
    return submit_job_path


def run_experiment(directory):
    rundir = os.path.join(directory, 'rundir')
    submit_job_path = write_submit_job_script(rundir)
    return check_call(['bash', submit_job_path])

