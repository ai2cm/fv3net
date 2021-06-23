import loaders.typing
import xarray as xr
from typing_extensions import runtime_checkable, Protocol


def test_is_mapper_function():
    def mock_mapper_function(data_path: str, other_arg: str) -> loaders.typing.Mapper:
        return {"key": xr.Dataset()}

    assert isinstance(
        mock_mapper_function, runtime_checkable(loaders.typing.MapperFunction)
    )


def test_extended_kwargs_signature_is_still_instance():
    @runtime_checkable
    class MyProtocol(Protocol):
        def __call__(self, arg: str, **kwargs):
            pass

    def my_function(arg: str, kwarg1: int = 0):
        pass

    assert isinstance(my_function, MyProtocol)


def test_extended_args_signature_is_still_instance():
    @runtime_checkable
    class MyProtocol(Protocol):
        def __call__(self, arg: str, *args):
            pass

    def my_function(arg: str, arg2: int = 0):
        pass

    assert isinstance(my_function, MyProtocol)
