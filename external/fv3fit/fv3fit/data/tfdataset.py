import dataclasses
from typing import Mapping, Sequence, Optional
import tensorflow as tf
from .base import TFDatasetLoader, register_tfdataset_loader
import dacite


@dataclasses.dataclass
class VariableConfig:
    unstacked_dims: Sequence[str]


@register_tfdataset_loader
@dataclasses.dataclass
class WindowedZarrLoader(TFDatasetLoader):
    """
    A tfdataset loader that loads directly from zarr and supports time windows.

    Attributes:
        data_path: path to zarr data
        window_size: number of timesteps to include in time windows
        default_variable_config: default configuration for variables
        variable_configs: configuration for variables by name
    """

    data_path: str
    window_size: int
    default_variable_config: VariableConfig
    variable_configs: Mapping[str, VariableConfig] = dataclasses.field(
        default_factory=dict
    )

    def get_data(
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
        # if local_download_path is given, cache on disk
        # using tfdataset.cache(local_download_path)
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, d: dict) -> "WindowedZarrLoader":
        return dacite.from_dict(
            data_class=cls, data=d, config=dacite.Config(strict=True)
        )
