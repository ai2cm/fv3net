import os
import pytest

from fv3fit._shared.io import dump, load, register, _Register


def test_Register_get_name():
    register = _Register()

    @register("mock")
    class Mock:
        pass

    mock = Mock()
    assert register._get_name(mock) == "mock"


def test_registering_twice_fails():
    register = _Register()

    @register("mock")
    class Mock:
        pass

    with pytest.raises(ValueError):

        @register("mock")
        class Mock2:
            pass


def test_register_dump_load(tmpdir):

    register = _Register()

    relative_path = "some_path"

    @register("mock1")
    class Mock1:
        def __init__(self, data):
            self.data = data

        @staticmethod
        def load(path: str):
            with open(os.path.join(path, relative_path)) as f:
                return Mock1(f.read())

        def dump(self, path: str):
            with open(os.path.join(path, relative_path), "w") as f:
                return f.write(self.data)

    m = Mock1(data="hello")
    register.dump(m, str(tmpdir))
    m_loaded = register.load(str(tmpdir))
    assert m.data == m_loaded.data


def test_external_shared_io_interface():
    assert isinstance(register, _Register)
    assert dump == register.dump
    assert load == register.load
