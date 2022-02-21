import argparse
from typing import Sequence
from loaders._config import BatchesLoader
from loaders._utils import SAMPLE_DIM_NAME
import yaml
import os.path
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Save a BatchesLoader configuration locally "
            "as a directory of netCDF files."
        )
    )
    parser.add_argument(
        "data_config",
        type=str,
        help="path of loaders.BatchesLoader training data yaml file",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="local directory to save data as numbered netCDF files",
    )
    parser.add_argument(
        "-n",
        "--variable-names",
        nargs="+",
        default=[],
        help=(
            "variable names to include in saved file, "
            "passed to BatchesLoader.load_batches"
        ),
    )
    return parser


def main(data_config: str, output_path: str, variable_names: Sequence[str]):
    with open(data_config, "r") as f:
        config = yaml.safe_load(f)
    loader = BatchesLoader.from_dict(config)
    logger.info("configuration loaded, creating batches object")
    batches = loader.load_batches(variables=variable_names)
    n_batches = len(batches)
    logger.info(f"batches object created, saving {n_batches} batches")
    for i, batch in enumerate(batches):
        out_filename = os.path.join(output_path, f"{i:05}.nc")
        logger.info(f"saving batch {i}")
        try:
            batch.to_netcdf(out_filename, engine="h5netcdf")
        except NotImplementedError:
            batch.reset_index(dims_or_levels=[SAMPLE_DIM_NAME]).to_netcdf(
                out_filename, engine="h5netcdf"
            )


if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
    )
    handler.setLevel(logging.INFO)
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    main(
        data_config=args.data_config,
        output_path=args.output_path,
        variable_names=args.variable_names,
    )
