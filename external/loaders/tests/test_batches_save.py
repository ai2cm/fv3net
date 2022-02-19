import loaders
import unittest.mock
from loaders.testing import mapper_context, batches_from_mapper_context
from loaders.batches import save
import tempfile
import yaml
import xarray as xr
import numpy as np


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


def test_save_stacked_data(tmpdir):
    with mapper_context(), batches_from_mapper_context():

        @loaders._config.mapper_functions.register
        def mock_mapper():
            return {
                "0": xr.Dataset(
                    data_vars={
                        "a": xr.DataArray(
                            np.zeros([10, 5, 5, 7]), dims=["dim0", "dim1", "dim2", "z"]
                        )
                    }
                )
            }

        loaders._config.batches_from_mapper_functions.register(
            loaders.batches_from_mapper
        )

        config_dict = {
            "function": "batches_from_mapper",
            "mapper_config": {"function": "mock_mapper", "kwargs": {}},
            "kwargs": {"unstacked_dims": ["z"], "variable_names": ["a"]},
        }
        with tempfile.NamedTemporaryFile() as tmpfile:
            with open(tmpfile.name, "w") as f:
                yaml.dump(config_dict, f)
            # only need to test a lack of crash for stacked data,
            # batch.to_netcdf being called is checked in another test
            save.main(data_config=tmpfile.name, output_path=tmpdir)
