import fv3fit.data
import dataclasses
import unittest.mock


def test_loader_from_dict():
    mock_result = unittest.mock.MagicMock()

    @fv3fit.data.register_tfdataset_loader
    @dataclasses.dataclass
    class MyLoaderSubclass(fv3fit.data.TFDatasetLoader):
        foo: int
        bar: float

        @classmethod
        def from_dict(cls, d):
            return mock_result

        def get_data(
            self, local_download_path, variable_names,
        ):
            raise NotImplementedError()

    config = {"foo": 1, "bar": 2.0}
    result = fv3fit.data.tfdataset_loader_from_dict(config)
    assert result is mock_result
