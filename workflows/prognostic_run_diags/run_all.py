import yaml
import subprocess
import time
import os

N_JOBS = 4
Q = []

with open("rundirs.yml") as f:
    rundirs = yaml.safe_load(f)
    

def name_output_bucket(rundir):
    return os.path.join(rundir, 'metrics.zarr')


def run_command(rundir, output):
    return subprocess.Popen(['python', '-m', 'fv3net.pipelines.savediags', rundir, output])
    

def run_argo(rundir, output):
    return subprocess.check_call(['argo', 'submit', '-p', 'run='+rundir, '-p', 'output='+output, 'argo.yaml'])


def run_in_shell(rundirs):
    while True and len(rundirs) > 0:
        if len(Q) <= N_JOBS:
            rundir = rundirs.pop()
            output = name_output_bucket(rundir)
            Q.append(run_command(rundir, output))
        else:
            time.sleep(1)
            running = []
            while len(Q) > 1:
                proc = Q.pop()
                if proc.poll() is None:
                    running.append(proc)
            Q.extend(running)


def submit_to_argo(rundirs):
    for rundir in rundirs:
        output = name_output_bucket(rundir)
        run_argo(rundir, output)


if __name__ == "__main__":
    submit_to_argo(rundirs)