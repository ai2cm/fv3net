import argparse
import subprocess
import wandb


train_timestamps = [
    "20160101.000000",
    "20160301.000000",
    "20160401.000000",
    "20160501.000000",
    "20160701.000000",
    "20160801.000000",
    "20161001.000000",
    "20161101.000000",
    "20161201.000000",
]


valid_timestamps = [
    "20160201.000000",
    "20160601.000000",
    "20160901.000000",
]

PROJECT = "2021-10-14-microphysics-emulation-paper"
BUCKET = "vcm-ml-scratch"
DATE = "2021-11-18"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    train_out = f"{base_url}/training_netcdfs/train"
    valid_out = f"{base_url}/training_netcdfs/test"

    command = ["gsutil", "-m", "cp"]

    for timestamp in train_timestamps:
        tag = f"{args.tag_prefix}-{timestamp}"
        nc_src = f"{base_url}/{tag}/artifacts/{timestamp}/netcdf_output/*.nc"
        dir_args = [nc_src, train_out]
        subprocess.check_call(command + dir_args)

    for timestamp in valid_timestamps:
        tag = f"{args.tag_prefix}-{timestamp}"
        nc_src = f"{base_url}/{tag}/artifacts/{timestamp}/netcdf_output/*.nc"
        dir_args = [nc_src, valid_out]
        subprocess.check_call(command + dir_args)

    artifact = wandb.Artifact("microphysics-training-data", type="training_netcdfs")
    artifact.add_reference(f"{base_url}/training_netcdfs")
    wandb.log_artifact(artifact)
