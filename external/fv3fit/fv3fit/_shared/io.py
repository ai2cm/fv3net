from typing import MutableMapping, Callable, Type, TypeVar, cast
import os
import fsspec
import warnings

from .predictor import Reloadable
from functools import partial

_NAME_PATH = "name"
_NAME_ENCODING = "UTF-8"

DEPCRECATED_NAMES = {"packed-keras": "007bc80046c29ae3e2a535689b5c68e46cf2c613"}

R = TypeVar("R", bound=Type[Reloadable])


class _Register:
    """Class to register new I/O names
    """

    def __init__(self) -> None:
        self._model_types: MutableMapping[str, Type[Reloadable]] = {}

    def __call__(self, name: str) -> Callable[[R], R]:
        if name in self._model_types:
            raise ValueError(
                f"{name} is already registered by {self._model_types[name]}."
            )
        else:
            return cast(Callable[[R], R], partial(self._register_class, name=name))

    def _register_class(self, cls: R, name: str) -> R:
        self._model_types[name] = cls
        return cls

    def _load_by_name(self, name: str, path: str) -> Reloadable:
        if name in DEPCRECATED_NAMES:
            last_valid_commit = DEPCRECATED_NAMES[name]
            raise ValueError(
                f"fv3fit models of name '{name}' are deprecated and can no longer be "
                f"loaded. '{name}' can be loaded by fv3net before {last_valid_commit}."
            )
        return self._model_types[name].load(path)

    def get_name(self, obj: Reloadable) -> str:
        return_name = None
        name_cls = None
        for name, cls in self._model_types.items():
            if isinstance(obj, cls):
                # always return the most specific class name / deepest subclass
                if name_cls is None or issubclass(cls, name_cls):
                    return_name = name
                    name_cls = cls
        if return_name is None:
            raise ValueError(
                f"{type(obj)} is not registered. "
                'Consider decorating with @fv3fit._shared.io.register("name")'
            )
        else:
            return return_name

    @staticmethod
    def _get_name_from_path(path: str) -> str:
        return fsspec.get_mapper(path)[_NAME_PATH].decode(_NAME_ENCODING).strip()

    def _dump_class_name(self, obj: Reloadable, path: str):
        mapper = fsspec.get_mapper(path)
        name = self.get_name(obj)
        mapper[_NAME_PATH] = name.encode(_NAME_ENCODING)

    def load(self, path: str) -> Reloadable:
        """Load a serialized Reloadable from `path`."""
        try:
            name = self._get_name_from_path(path)
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

    def dump(self, obj: Reloadable, path: str):
        """Dump a Reloadable to a path"""
        self._dump_class_name(obj, path)
        obj.dump(path)


register = _Register()
dump = register.dump
load = register.load
