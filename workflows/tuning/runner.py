import argparse
import subprocess
import yaml
import os
import sherpa
import time

TRIAL_PY_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', help='configuration yaml location', type=str)
    parser.add_argument('outdir', help='output location for scheduler and database files', type=str)
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    return parser.parse_args()


def get_study(outdir, config):
    with open(args.configfile, 'r') as f:
        config = yaml.safe_load(f)
    algorithm_name = config["algorithm"]["name"]
    if not hasattr(sherpa.algorithms, algorithm_name):
        raise ValueError(f"No sherpa algorithm {algorithm_name} exists, is there a typo?")
    algorithm = getattr(sherpa.algorithms, algorithm_name)(
        *config["algorithm"].get("args", []),
        **config["algorithm"].get("kwargs", {})
    )
    parameters = []
    for parameter_config in config["parameters"]:
        parameters.append(sherpa.core.Parameter.from_dict(parameter_config))
    return sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        output_dir=outdir,
    )


class TrialJob:

    RUNNING = 0
    COMPLETED = 1
    FAILED = 2

    def __init__(self, trial):
        self.trial = trial

    def start(self):
        self._process = subprocess.Popen()

    @property
    def status(self):
        pass


def run_study(study, max_jobs=1):
    waiting_jobs = [TrialJob(trial) for trial in study]
    running_jobs = []
    while len(waiting_jobs) > 0:
        while (len(running_jobs) < max_jobs) and len(waiting_jobs) > 0:
            job = waiting_jobs.pop()
            job.start()
            running_jobs.append(job)
        time.sleep(1)
        for job in running_jobs:
            if job.status != TrialJob.RUNNING:
                pass
    for trial in study:
        study.add_observation(trial, objective=1)
        study.finalize(trial)
    study.save()


if __name__ == '__main__':
    args = parse_args()
    with open(args.configfile, 'r') as f:
        config = yaml.safe_load(f)
    study = get_study(args.outdir, config)
    run_study(study, max_jobs=args.max_concurrent)
    print(f"Best result: {study.get_best_result()}")
