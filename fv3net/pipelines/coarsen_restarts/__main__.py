import argparse
import logging
from .pipeline import run

if __name__ == "__main__":

    logger = logging.getLogger("CoarsenTimesteps")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gcs-src-dir",
        type=str,
        required=True,
        help="Full GCS path to input data for downloading timesteps.",
    )
    parser.add_argument(
        "--gcs-dst-dir",
        type=str,
        help=(
            "Full GCS path to output coarsened timestep data. Defaults to input path"
            "with target resolution appended as a directory"
        ),
    )
    parser.add_argument(
        "--gcs-grid-spec-path",
        type=str,
        required=True,
        help=(
            "Full path with file wildcard 'grid_spec.tile*.nc' to select grid spec"
            " files with same resolution as the source data."
        ),
    )
    parser.add_argument(
        "--source-resolution",
        type=int,
        required=True,
        help="Source data cubed-sphere grid resolution.",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        required=True,
        help="Target coarsening resolution to output.",
    )
    parser.add_argument(
        "--add-target-subdir",
        type=bool,
        required=False,
        default=True,
        help=(
            "Add subdirectory with C{target-resolution} to the destination directory."
            " If --gcs-dst-dir is not specified, then the subdir is automatically added."
        ),
    )

    args, pipeline_args = parser.parse_known_args()
    run(args=args, pipeline_args=pipeline_args)
