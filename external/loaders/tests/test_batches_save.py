import loaders
import unittest.mock
from loaders.testing import mapper_context, batches_from_mapper_context
from loaders.batches import save
import tempfile
import yaml
import xarray as xr


def test_save(tmpdir):
    """
    Test that the batches described by the config have .to_netcdf(filename)
    called on each of them, with the expected filenames.
    """
    mock_batches = [unittest.mock.MagicMock(spec=xr.Dataset) for _ in range(3)]
    filenames = [f"{tmpdir}/00000.nc", f"{tmpdir}/00001.nc", f"{tmpdir}/00002.nc"]
    with mapper_context(), batches_from_mapper_context():

        @loaders._config.mapper_functions.register
        def mock_mapper():
            return None

        @loaders._config.batches_from_mapper_functions.register
        def mock_batches_from_mapper(
            mapping_function, variable_names, mapping_kwargs=None,
        ):
            return mock_batches

        config_dict = {
            "function": "mock_batches_from_mapper",
            "mapper_config": {"function": "mock_mapper", "kwargs": {}},
            "kwargs": {},
        }
        with tempfile.NamedTemporaryFile() as tmpfile:
            with open(tmpfile.name, "w") as f:
                yaml.dump(config_dict, f)
            save.main(data_config=tmpfile.name, output_path=tmpdir)
        for filename, batch in zip(filenames, mock_batches):
            batch.to_netcdf.assert_called_once_with(filename, engine="h5netcdf")
