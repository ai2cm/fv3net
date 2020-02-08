import argparse
import os

import submit_utils

PWD = os.path.dirname(os.path.abspath(__file__))
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(PWD, "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(PWD, "diag_table")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_gcs_path",
        type=str,
        required=True,
        help="Source input restart files for one-step simulations",
    )
    parser.add_argument(
        "output_gcs_path",
        type=str,
        required=True,
        help="Output destination for one-step configurations and simulation output"
    )
