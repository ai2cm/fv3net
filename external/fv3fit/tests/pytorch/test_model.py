from fv3fit.pytorch import PytorchModel, _pack_to_tensor
from torch import nn
import fv3fit
import numpy as np
import torch
import xarray as xr


def same_state(model1: nn.Module, model2: nn.Module) -> bool:
    """
    Check if two models have the same state.
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    if state1.keys() != state2.keys():
        return False
    for key in state1.keys():
        if not torch.equal(state1[key], state2[key]):
            return False
    return True


def test_pytorch_model_dump_load(tmpdir):
    scaler = fv3fit.StandardScaler()
    n_features = 2
    n_samples = 10
    state_variables = ["u"]
    data = np.random.uniform(low=-1, high=1, size=(n_samples, n_features))
    scaler.fit(data)
    model = PytorchModel(
        state_variables=state_variables,
        model=nn.Linear(n_features, n_features),
        scalers={"u": scaler},
    )
    model.dump(str(tmpdir))
    reloaded_model = PytorchModel.load(str(tmpdir))
    assert model.state_variables == reloaded_model.state_variables
    assert same_state(model.model, reloaded_model.model)
    assert model.scalers == reloaded_model.scalers


def test__pack_to_tensor_one_var_align_times():
    # easier to understand these values, but they only test time alignment
    ntime, ntiles, nx, ny, nz = 11, 6, 8, 8, 2
    data = np.zeros((ntime, ntiles, nx, ny, nz))
    data[:] = np.arange(0, ntime)[:, None, None, None, None]
    _helper_test_pack_to_tensor_one_var(data)


def test__pack_to_tensor_one_var():
    # random values test spatial alignment as well
    ntime, ntiles, nx, ny, nz = 11, 6, 8, 8, 2
    data = np.random.uniform(low=-1, high=1, size=(ntime, ntiles, nx, ny, nz))
    _helper_test_pack_to_tensor_one_var(data)


def _helper_test_pack_to_tensor_one_var(data):
    ds = xr.Dataset(
        data_vars={"u": xr.DataArray(data, dims=["time", "tile", "x", "y", "z"])}
    )
    scaler = fv3fit.StandardScaler()
    scaler.fit(data)
    # disable normalization so we can compare values directly
    scaler.mean[:] = 0.0
    scaler.std[:] = 1.0
    tensor = _pack_to_tensor(
        ds=ds, timesteps=2, state_variables=["u"], scalers={"u": scaler}
    )
    assert tensor.shape[0] == 5
    # check end of a window
    np.testing.assert_almost_equal(tensor[2, -1, :], data[6, :])
    # check a full window
    np.testing.assert_almost_equal(tensor[2, :], data[4:7, :])
