import argparse
import os
import subprocess
import wandb


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
            "<prefix><month>/artifacts/*/netcdf_output"
        )
    )
    parser.add_argument("tag_prefix", help="Common URL prefix for training data runs")
    parser.add_argument("output_prefix", help="Output prefix for training data")
    args = parser.parse_args()

    run = wandb.init(
        job_type="netcdf-gather", project="microphysics-emulation", entity="ai2cm"
    )
    wandb.config.update(args)
    wandb.config["env"] = {"COMMIT_SHA": os.getenv("COMMIT_SHA", "")}

    assert args.tag_prefix.startswith("gs://")
    assert args.output_prefix.startswith("gs://")

    output_prefix = args.output_prefix.rstrip("/")

    train_out = output_prefix + "/" + "train"
    valid_out = output_prefix + "/" + "test"

    command = ["gsutil", "-m", "cp"]

    for timestamp in train_timestamps:
        tag = f"{args.tag_prefix}{timestamp}"
        nc_src = f"{tag}/artifacts/*/netcdf_output/*.nc"
        dir_args = [nc_src, train_out]
        subprocess.check_call(command + dir_args)

    for timestamp in valid_timestamps:
        tag = f"{args.tag_prefix}{timestamp}"
        nc_src = f"{tag}/artifacts/*/netcdf_output/*.nc"
        dir_args = [nc_src, valid_out]
        subprocess.check_call(command + dir_args)

    artifact = wandb.Artifact("microphysics-training-data", type="training_netcdfs")
    artifact.add_reference(output_prefix, checksum=False)
    wandb.log_artifact(artifact)
