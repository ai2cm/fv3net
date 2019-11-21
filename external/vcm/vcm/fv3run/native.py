import os
from subprocess import check_call
from .common import write_submit_job_script

FV3_EXE = os.getenv('FV3_EXE', '/FV3/fv3.exe')



def run_experiment(directory):
    rundir = os.path.join(directory, 'rundir')
    submit_job_path = write_submit_job_script(rundir)
    return check_call(['bash', submit_job_path])

