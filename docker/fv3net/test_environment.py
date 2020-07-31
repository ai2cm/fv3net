import tensorflow, fv3net, fv3fit, apache_beam  # noqa
import subprocess
import sys
import os


def test_gsutil_not_from_conda():
    gsutil_path = subprocess.check_output(["which", "gsutil"]).decode("UTF-8")
    conda_bin_dir = os.path.dirname(sys.executable)
    if gsutil_path.startswith(conda_bin_dir):
        raise AssertionError(f"gsutil found at {gsutil_path}")


def test_touch_config_file():
    """this test will catch errors related to permissions issues
    or google authentication"""
    google_config = "/home/jovyan/.config/gcloud/test_write_file"
    with open(google_config, "w") as f:
        f.write("Hello")
        
        
def test_dataflow_environment():
    '''Ensure that the dataflow software packages can be installed'''
    subprocess.check_call(["/home/jovyan/fv3net/dataflow.sh", "check"])
