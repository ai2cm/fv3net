import argparse
from fv3net.pipelines.create_training_data.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs-input-data-path",
        type=str,
        required=True,
        help="Location of input data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )
    parser.add_argument(
        "--gcs-output-data-dir",
        type=str,
        required=True,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default="gs://vcm-ml-data",
        help="Google Cloud Storage bucket name.",
    )
    parser.add_argument(
        "--gcs-project",
        type=str,
        default="vcm-ml",
        help="Project name for google cloud.",
    )
    parser.add_argument(
        "--mask-to-surface-type",
        type=str,
        default=None,
        help="Mask to surface type in ['sea', 'land', 'seaice'].",
    )
    parser.add_argument(
        "--timesteps-per-output-file",
        type=int,
        default=2,
        help="Number of consecutive timesteps to calculate features/targets for in "
        "a single process and save to output file."
        "When the full output is shuffled at the data generator step, these"
        "timesteps will always be in the same training data batch.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of batches to save as training set. "
        "Output zarr files will be saved in either 'train' or 'test' subdir of "
        "gcs-output-data-dir",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed used when shuffling data batches before test/train selection",
    )
    args, pipeline_args = parser.parse_known_args()

    """Main function"""
    run(args=args, pipeline_args=pipeline_args)
