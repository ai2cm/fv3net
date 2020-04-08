import fv3config
import argparse
import yaml
import string
import secrets
import fv3net.pipelines.kube_jobs as kube_jobs

DEFAULT_JOB_PREFIX = "nudge_to_high_res"
ORCHESTRATOR_JOBS_LABEL = "orchestrator-jobs"

KUBERNETES_DEFAULT = {
    "cpu_count": 6,
    "memory_gb": 3.6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="google cloud storage location of fv3config yaml configuration file")
    parser.add_argument('runfile', type=str, help="google cloud storage location of python model run file")
    parser.add_argument('output_url', type=str, help="google cloud storage location to upload the resulting run directory")
    parser.add_argument('docker_image', type=str, help="docker image with fv3gfs-python to run the model")
    parser.add_argument('--job_prefix', type=str, default=DEFAULT_JOB_PREFIX, help="prefix to use in creating the job name")
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="if given, do not wait for the k8s job to complete",
    )
    return parser.parse_args()


def random_tag(length):
    use_chars = string.ascii_lowercase + string.digits
    return "".join([secrets.choice(use_chars) for i in range(length)])


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    kube_opts = KUBERNETES_DEFAULT.copy()
    job_name = '{}-{}'.format(args.job_prefix, random_tag(8))
    job_label = {ORCHESTRATOR_JOBS_LABEL: job_name}
    fv3config.run_kubernetes(
        config_location=args.config,
        outdir=args.output_url,
        jobname=job_name,
        docker_image=args.docker_image,
        job_labels=job_label,
        **kube_opts,
    )

    if not args.detach:
        successful, _ = kube_jobs.wait_for_complete(job_label)
        kube_jobs.delete_job_pods(successful)
