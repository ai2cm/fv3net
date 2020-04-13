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
        "gcs_output_data_dir",
        type=str,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )
    parser.add_argument(
        "--timesteps-per-output-file",
        type=int,
        default=1,
        help="Number of consecutive timesteps to calculate features/targets for in "
        "a single process and save to output file."
        "When the full output is shuffled at the data generator step, these"
        "timesteps will always be in the same training data batch.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Fraction of batches to save as training set. "
        "Output zarr files will be saved in either 'train' or 'test' subdir of "
        "gcs-output-data-dir",
    )

    args, pipeline_args = parser.parse_known_args()
    with open(args.variable_namefile, "r") as stream:
        names = yaml.safe_load(stream)
    run(args=args, pipeline_args=pipeline_args, names=names)
