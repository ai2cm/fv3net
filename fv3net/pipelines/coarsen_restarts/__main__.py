import os
import argparse
import logging
from .pipeline import run

if __name__ == "__main__":

    logger = logging.getLogger("CoarsenTimesteps")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    # TODO Remove "gcs" from these names. It's not longer true
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
    logging.basicConfig(level=logging.DEBUG)

    args, pipeline_args = parser.parse_known_args()

    gridspec_path = os.path.join(args.gcs_grid_spec_path, "grid_spec")

    output_dir_prefix = args.gcs_dst_dir
    if args.add_target_subdir:
        output_dir_prefix = os.path.join(
            output_dir_prefix, f"C{args.target_resolution}"
        )

    # TODO change this CLI. Why use 2 args (which might not eveny divide) when
    # 1 will do?
    factor = args.source_resolution // args.target_resolution

    prefix = args.gcs_dst_dir
    tmp_timestep_dir = os.path.join(prefix, "local_fine_dir")

    run(
        gridspec_path,
        args.gcs_src_dir,
        output_dir_prefix,
        factor,
        pipeline_args=pipeline_args,
    )
