import os
import subprocess


def authenticate():
    try:
        credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError:
        pass
    else:
        subprocess.check_call(
            ["gcloud", "auth", "activate-service-account", "--key-file", credentials]
        )


def upload_dir(d, dest):
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", "-e", d, dest])


def download_directory(dir_, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", dir_, dest])


def cp(source, destination):
    subprocess.check_call(["gsutil", "cp", source, destination])
