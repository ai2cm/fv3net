import argparse
import yaml

from .pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gcs_input_data_path",
        type=str,
        help="Location of input data in Google Cloud Storage bucket. "
        "Don't include bucket or big.zarr in path.",
    )
    parser.add_argument(
        "diag_c48_path",
        type=str,
        help="Directory containing diagnostic zarr directory coarsened from C384 "
        "for features (SHF, LHF, etc.) that are not saved in restarts.",
    )
    parser.add_argument(
        "variable_namefile",
        type=str,
        default=None,
        help="yaml file for providing data variable names",
    )
    parser.add_argument(
        "timesteps_file",
        type=str,
        help="File containing list of paired timesteps in test & train sets."
    )
    parser.add_argument(
        "gcs_output_data_dir",
        type=str,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )


    args, pipeline_args = parser.parse_known_args()
    with open(args.variable_namefile, "r") as f:
        names = yaml.safe_load(f)
    with open(args.timesteps_file, "r") as f:
        timesteps = yaml.safe_load(f)
    run(args=args, pipeline_args=pipeline_args, names=names, timesteps=timesteps)
