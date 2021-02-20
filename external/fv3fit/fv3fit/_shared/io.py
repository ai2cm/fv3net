from typing import MutableMapping, Callable
import os
import fsspec
import warnings

from .predictor import Predictor, Estimator
from functools import partial

_NAME_PATH = "name"
_NAME_ENCODING = "UTF-8"




class _Register:
    """Class to register new I/O names
    """

    def __init__(self) -> None:
        self._model_types: MutableMapping[str, type] = {}

    def __call__(self, name: str) -> Callable[[type], type]:
        if name in self._model_types:
            raise ValueError(
                f"{name} is already registered by {self._model_types[name]}."
            )
        else:
            return partial(self._register_class, name=name)

    def _register_class(self, cls: type, name: str) -> type:
        self._model_types[name] = cls
        return cls

    def _load_by_name(self, name: str, path: str) -> Predictor:
        return self._model_types[name].load(path)  # type: ignore

    def _get_name(self, obj: Predictor) -> str:
        for name, cls in self._model_types.items():
            if isinstance(obj, cls):
                return name
        raise ValueError(
            f"{type(obj)} is not registered. "
            'Consider decorating with @fv3fit._shared.io.register("name")'
        )

    @staticmethod
    def _get_mapper_name(path: str) -> str:
        return fsspec.get_mapper(path)[_NAME_PATH].decode(_NAME_ENCODING)

    def _set_mapper_name(self, obj: Predictor, path: str):
        mapper = fsspec.get_mapper(path)
        name = self._get_name(obj)
        mapper[_NAME_PATH] = name.encode(_NAME_ENCODING)

    def load(self, path: str) -> Predictor:
        """Load a serialized model from `path`."""
        try:
            name = self._get_mapper_name(path)
        except KeyError as e:
            # backwards compatibility
            warnings.warn(
                "Model type is not located at "
                f"{os.path.join(path, _NAME_PATH)}. "
                "Trying all known models one-by-one.",
                UserWarning,
            )
            for name in self._model_types:
                try:
                    return self._load_by_name(name, path)
                except:  # noqa
                    pass
            raise e
        else:
            return self._load_by_name(name, path)

    def dump(self, obj: Estimator, path: str):
        """Dump an Estimator to a path"""
        self._set_mapper_name(obj, path)
        obj.dump(path)


register = _Register()
dump = register.dump
load = register.load
