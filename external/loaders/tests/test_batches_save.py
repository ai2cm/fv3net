import loaders
import unittest.mock
from loaders.testing import mapper_context, batches_from_mapper_context
from loaders.batches import save
import tempfile
import yaml
import xarray as xr
import numpy as np
import os
import pytest


def test_save(tmpdir):
    """
    Test that the batches described by the config have .to_netcdf(filename)
    called on each of them, with the expected filenames.
    """
    mock_batches = [unittest.mock.MagicMock(spec=xr.Dataset) for _ in range(3)]
    filenames = [f"{tmpdir}/00000.nc", f"{tmpdir}/00001.nc", f"{tmpdir}/00002.nc"]
    with mapper_context(), batches_from_mapper_context() as mock_batches_from_mapper:
        mock_batches_from_mapper.return_value = mock_batches

        @loaders._config.mapper_functions.register
        def mock_mapper():
            return None

        config_dict = {
            "mapper_config": {"function": "mock_mapper", "kwargs": {}},
        }
        with tempfile.NamedTemporaryFile() as tmpfile:
            with open(tmpfile.name, "w") as f:
                yaml.dump(config_dict, f)
            save.main(data_config=tmpfile.name, output_path=tmpdir, variable_names=[])
        for filename, batch in zip(filenames, mock_batches):
            batch.to_netcdf.assert_called_once_with(filename, engine="h5netcdf")


@pytest.mark.slow
def test_save_stacked_data(tmpdir):
    with mapper_context():

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

        config_dict = {
            "mapper_config": {"function": "mock_mapper", "kwargs": {}},
            "unstacked_dims": ["z"],
            "variable_names": ["a"],
        }
        with tempfile.NamedTemporaryFile() as tmpfile:
            with open(tmpfile.name, "w") as f:
                yaml.dump(config_dict, f)
            # only need to test a lack of crash for stacked data,
            # batch.to_netcdf being called is checked in another test
            save.main(data_config=tmpfile.name, output_path=tmpdir, variable_names=[])
            ds = xr.open_dataset(os.path.join(str(tmpdir), "00000.nc"))
            assert "a" in ds.data_vars
            assert ds.a.dims == ("_fv3net_sample", "z")


@pytest.mark.slow
def test_save_raises_on_missing_variable(tmpdir):
    with mapper_context():

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

        config_dict = {
            "mapper_config": {"function": "mock_mapper", "kwargs": {}},
            "unstacked_dims": ["z"],
            "variable_names": ["a"],
        }
        with tempfile.NamedTemporaryFile() as tmpfile:
            with open(tmpfile.name, "w") as f:
                yaml.dump(config_dict, f)
            # only need to test a lack of crash for stacked data,
            # batch.to_netcdf being called is checked in another test
            with pytest.raises(KeyError):
                save.main(
                    data_config=tmpfile.name, output_path=tmpdir, variable_names=["b"]
                )
