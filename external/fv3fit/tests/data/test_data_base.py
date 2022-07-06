import fv3fit.data
import dataclasses


def test_loader_from_dict():
    @dataclasses.dataclass
    class MyLoaderSubclass(fv3fit.data.TFDatasetLoader):
        foo: int
        bar: float

        def get_data(
            self, local_download_path, variable_names,
        ):
            raise NotImplementedError()

    config = {"foo": 1, "bar": 2.0}
    result = fv3fit.data.TFDatasetLoader.from_dict(config)
    assert isinstance(result, MyLoaderSubclass)
    assert result.foo == 1
    assert result.bar == 2.0
