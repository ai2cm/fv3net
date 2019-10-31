import subprocess

PROJECT_ID = 'vcm-ml'
REGION = 'us-central1'
BUCKET = "gs://vcm-ml-data/"
LOGS = 'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/dsub-logs/'
IMAGE = 'us.gcr.io/vcm-ml/fv3net'

def get_timesteps():
    b = subprocess.check_output(['bash', 'list_timesteps_to_run.sh'])
    s = b.decode('UTF-8')
    return s.strip().split()


def log_file(time):
    return LOGS.format(timestep=time)


def get_command_for_docker(time):
    return f"'python3 k8s-workflows/scale-snakemake/run_snakemake_for_timestep.py {time}'"


def command(time):
    return [
        'dsub',
        '--provider', 'google-v2',
        '--project', PROJECT_ID,
        '--logging', log_file(time),
        '--regions', REGION,
        '--image', IMAGE,
        '--command', get_command_for_docker(time)
    ]


def submit_job_time(time):
    return subprocess.check_call(command(time))


a_time = '20160801.003000'
submit_job_time(a_time)
