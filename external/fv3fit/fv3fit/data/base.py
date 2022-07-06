import abc
import tensorflow as tf
from typing import Optional, Sequence
import dacite


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

    @classmethod
    def from_dict(cls, kwargs) -> "TFDatasetLoader":
        try:
            return dacite.from_dict(data_class=cls, data=kwargs)
        except (TypeError, AttributeError):
            pass
        for subclass in cls.__subclasses__():
            print(subclass)
            try:
                # if the subclass defines its own from_dict use that,
                # otherwise use dacite
                if (
                    hasattr(subclass, "from_dict")
                    and subclass.from_dict is not cls.from_dict
                ):
                    return subclass.from_dict(kwargs)
                else:
                    return dacite.from_dict(data_class=subclass, data=kwargs)
            except (
                TypeError,
                ValueError,
                AttributeError,
                dacite.exceptions.MissingValueError,
            ):
                pass
        raise ValueError("invalid TFDatasetLoader dictionary")
