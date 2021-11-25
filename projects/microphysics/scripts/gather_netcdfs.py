import argparse
import subprocess
import wandb
from config import PROJECT, BUCKET


# Initial condition timesteps to gather for training data
train_timestamps = ["1", "3", "4", "5", "7", "8", "10", "11", "12"]


# Initial condition timesteps to gather for testing data
valid_timestamps = ["2", "6", "9"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script gathers netcdfs from a set of training runs"
            " and uploads them to train/test groups on GCS. Runs are"
            " assumed to be in "
            "gs://<BUCKET>/<PROJECT>/<RUN_DATE>/<TAG>/artifacts/<TIMESTAMP>/netcdf_output"  # noqa E501
            " format to be gathered.  The <RUN_DATE> is the date the job"
            " was performed and <TAG> the unique project name."
        )
    )
    parser.add_argument(
        "run_date", help="date string when training was run, e.g., 2021-11-18"
    )
    parser.add_argument("tag_prefix", help="Common tag prefix for training data runs")

    args = parser.parse_args()

    run = wandb.init(
        job_type="netcdf-gather", project="microphysics-emulation", entity="ai2cm"
    )
    wandb.config.update(args)

    base_url = f"gs://{BUCKET}/{PROJECT}/{args.run_date}"
    train_out = f"{base_url}/{args.tag_prefix}-training_netcdfs/train"
    valid_out = f"{base_url}/{args.tag_prefix}-training_netcdfs/test"

    command = ["gsutil", "-m", "cp"]

    for timestamp in train_timestamps:
        tag = f"{args.tag_prefix}-{timestamp}"
        nc_src = f"{base_url}/{tag}/artifacts/*/netcdf_output/*.nc"
        dir_args = [nc_src, train_out]
        subprocess.check_call(command + dir_args)

    for timestamp in valid_timestamps:
        tag = f"{args.tag_prefix}-{timestamp}"
        nc_src = f"{base_url}/{tag}/artifacts/*/netcdf_output/*.nc"
        dir_args = [nc_src, valid_out]
        subprocess.check_call(command + dir_args)

    artifact = wandb.Artifact("microphysics-training-data", type="training_netcdfs")
    artifact.add_reference(f"{base_url}/training_netcdfs")
    wandb.log_artifact(artifact)
