import os
import subprocess

import backoff


class GSUtilResumableUploadException(Exception):
    pass


def authenticate():
    try:
        credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError:
        pass
    else:
        subprocess.check_call(
            ["gcloud", "auth", "activate-service-account", "--key-file", credentials]
        )


@backoff.on_exception(backoff.expo, GSUtilResumableUploadException, max_tries=3)
def upload_dir(d, dest):
    try:
        subprocess.check_output(
            ["gsutil", "-m", "rsync", "-r", "-e", d, dest], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        if "ResumableUploadException" in e.output:
            raise GSUtilResumableUploadException()
        else:
            raise e


def download_directory(dir_, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", dir_, dest])


def cp(source, destination):
    subprocess.check_call(["gsutil", "cp", source, destination])
