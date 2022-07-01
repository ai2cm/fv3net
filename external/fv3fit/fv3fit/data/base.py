import abc
import tensorflow as tf
from typing import Optional, Sequence


class TFDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def get_data(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables
        """
        ...
