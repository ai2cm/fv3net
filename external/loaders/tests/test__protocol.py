from typing_extensions import runtime_checkable, Protocol


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
