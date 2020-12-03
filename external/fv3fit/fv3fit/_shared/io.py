from typing import Mapping, Callable
import os
import fsspec
import warnings

from .predictor import Predictor
from functools import partial

_NAME_PATH = "name"
_NAME_ENCODING = "UTF-8"


class _Register:
    """Class to register new I/O names
    """

    def __init__(self) -> None:
        self._model_types: Mapping[str, Predictor] = {}

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

    def _get_class(self, name: str) -> Predictor:
        return self._model_types[name]

    def _get_name(self, obj: Predictor) -> str:
        for name, cls in self._model_types.items():
            if isinstance(obj, cls):
                return name

    @staticmethod
    def _get_mapper_name(path: str) -> str:
        return fsspec.get_mapper(path)[_NAME_PATH].decode(_NAME_ENCODING)

    def _set_mapper_name(self, obj: Predictor, path: str):
        mapper = fsspec.get_mapper(path)
        name = self._get_name(obj)
        mapper[_NAME_PATH] = name.encode(_NAME_ENCODING)
        
    def load(self, path: str) -> object:
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
            for cls in self._model_types.values():
                try:
                    return cls.load(path)
                except: # noqa
                    pass
            raise e
        else:
            return self._get_class(name).load(path)

    def dump(self, obj: Predictor, path: str):
        """Dump a predictor to a path"""
        self._set_mapper_name(obj, path)
        obj.dump(path)


register = _Register()
dump = register.dump
load = register.load
