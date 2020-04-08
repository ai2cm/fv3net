import argparse
import logging
from .pipeline import run

if __name__ == "__main__":

    logger = logging.getLogger("CoarsenTimesteps")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "gcs_src_dir",
        type=str,
        help="Full GCS path to input data for downloading timesteps.",
    )
    parser.add_argument(
        "gcs_grid_spec_path",
        type=str,
        help=(
            "Full path with file wildcard 'grid_spec.tile*.nc' to select grid spec"
            " files with same resolution as the source data."
        ),
    )
    parser.add_argument(
        "source_resolution", type=int, help="Source data cubed-sphere grid resolution.",
    )
    parser.add_argument(
        "target_resolution", type=int, help="Target coarsening resolution to output.",
    )
    parser.add_argument(
        "gcs_dst_dir",
        type=str,
        help=(
            "Full GCS path to output coarsened timestep data. Defaults to input path"
            "with target resolution appended as a directory"
        ),
    )
    parser.add_argument(
        "--add-target-subdir",
        action="store_true",
        help=(
            "Add subdirectory with C{target-resolution} to the specified "
            "destination directory. "
        ),
    )

    args, pipeline_args = parser.parse_known_args()
    print(args)
    run(args=args, pipeline_args=pipeline_args)
