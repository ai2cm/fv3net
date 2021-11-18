import argparse
import subprocess
import wandb

from fv3net.artifacts.resolve_url import resolve_url


training_tags = [
    "test-create-training-20160101.000000",
    "test-create-training-20160301.000000",
    # "create-training-20160401.000000",
    # "create-training-20160501.000000",
    # "create-training-20160701.000000",
    # "create-training-20160801.000000",
    # "create-training-20161001.000000",
    # "create-training-20161101.000000",
    # "create-training-20161201.000000",
]


validation_tags = [
    "test-create-training-20160201.000000",
    # "create-training-20160601.000000",
    # "create-training-20160901.000000",
]

PROJECT = "2021-10-14-microphysics-emulation-paper"
BUCKET = "vcm-ml-scratch"
DATE = "2021-11-18"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_date",
        help="date string when training was run, e.g., 2021-11-18"
    )

    args = parser.parse_args()

    base_url = f"gs://{BUCKET}/{PROJECT}/{args.run_date}"
    train_out = f"{base_url}/training_netcdfs/train"
    valid_out = f"{base_url}/training_netcdfs/test"

    command = ["gsutil", "-m", "cp"]

    for tag in training_tags:
        ic_timestamp = tag.split("-")[-1]
        nc_src = f"{base_url}/{tag}/artifacts/{ic_timestamp}/netcdf_output/*"
        dir_args = [nc_src, train_out]
        subprocess.check_call(command + dir_args)

    for tag in validation_tags:
        ic_timestamp = tag.split("-")[-1]
        nc_src = f"{base_url}/{tag}/artifacts/{ic_timestamp}/netcdf_output/*"
        dir_args = [nc_src, valid_out]
        subprocess.check_call(command + dir_args)
