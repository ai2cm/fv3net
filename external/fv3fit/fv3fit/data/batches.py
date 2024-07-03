import dataclasses
import os
from typing import Optional, Sequence
import loaders
import loaders.typing
from loaders.batches.save import save_batches
from fv3fit.tfdataset import tfdataset_from_batches
import tensorflow as tf
import logging
from .base import TFDatasetLoader, register_tfdataset_loader

logger = logging.getLogger(__name__)


@register_tfdataset_loader
@dataclasses.dataclass
class FromBatches(TFDatasetLoader):

    batches_loader: loaders.BatchesLoader

    @classmethod
    def from_dict(cls, d: dict) -> "FromBatches":
        return FromBatches(loaders.BatchesLoader.from_dict(d))

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        batches = self.batches_loader.load_batches(variables=variable_names)
        if local_download_path is not None:
            batches = cache(batches, local_download_path, variable_names)
        batches = loaders.batches.shuffle(batches)
        return tfdataset_from_batches(batches)


def cache(
    batches: loaders.typing.Batches,
    local_download_path: str,
    variable_names: Sequence[str],
) -> loaders.typing.Batches:
    logger.info("saving batches data to %s", local_download_path)
    os.makedirs(local_download_path, exist_ok=True)
    save_batches(batches, output_path=local_download_path, num_jobs=4)
    return loaders.batches_from_netcdf(
        path=local_download_path, variable_names=variable_names
    )
