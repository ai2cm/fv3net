import argparse
import fsspec
import yaml
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List

from loaders.mappers import open_phys_emu_training
from loaders.batches import batches_from_mapper

"""
This script facilitates subsetting the training data batches into much
smaller datasets for NN training.  Uses the mappers.open_phys_emu_training
but probably could be expanded to use the general batch opening infrastructure
and our ML configurations.

Batch is defined as what the mapper loads for a single item.
"""


logger = logging.getLogger(__name__)


@dataclass
class SubsetConfig:
    """
    Class for subsampling job configuration.
    
    source_path: Source all-physics emulation training openable by mp.open_phys_emu_training
    destination_path: Destination for subsetted netcdf files
    init_times: Target initialization times to include in the dataset (used by 
        open_phys_emu_training)
    variables: Variables to include in the subset dataset
    subsample_size: Number of samples to draw from each batch
    num_workers: Number of threads to use for subsampling
    """
    source_path: str
    destination_path: str
    init_times: List[str]
    variables: List[str]
    subsample_size: int = 2560
    num_workers: int = 10


def subsample_batch(batches, item_idx):
    data = batches[item_idx]
    return item_idx, data


def run_subsample_threaded(num_workers, outdir, batches, template="window_{idx:04d}.nc"):

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(subsample_batch, batches, i)
            for i in range(len(batches))
        ]

        for future in as_completed(futures):
            item_idx, data = future.result()
            out_path = os.path.join(outdir, template.format(idx=item_idx))
            data.reset_index("sample").to_netcdf(out_path)
            logger.info(f"Completed subsetting item #{item_idx}")

    logger.info("Subsetting complete")


def save_to_destination(source, destination):

    fs = fsspec.get_fs_token_paths(destination)[0]
    fs.makedirs(destination, exist_ok=True)
    fs.put(source, destination, recursive=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Configuraiton description for the subsetting job"
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config_yaml = yaml.safe_load(f)

    config = SubsetConfig(**config_yaml)

    data_mapping = open_phys_emu_training(config.source_path, config.init_times,
        dataset_names=["emulator_inputs.zarr", "physics_tendencies.zarr"]
    )
    batches = batches_from_mapper(
        data_mapping, config.variables, subsample_size=config.subsample_size
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        with open(os.path.join(tmpdir, "subset_config.yaml"), "w") as f:
            f.write(yaml.dump(asdict(config)))
        
        run_subsample_threaded(config.num_workers, tmpdir, batches)
        save_to_destination(tmpdir, config.destination_path)
