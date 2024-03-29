from fv3fit.pytorch.training_loop import AutoregressiveTrainingConfig
import numpy as np
import xarray as xr
from typing import Sequence
from fv3fit.pytorch.graph import (
    GraphHyperparameters,
    train_graph_model,
    MPGraphUNetConfig,
    GraphUNetConfig,
)
from fv3fit.tfdataset import iterable_to_tfdataset
import collections
import os
import pytest
from fv3fit.pytorch.optimizer import OptimizerConfig
import fv3fit


def get_tfdataset(nsamples, nbatch, ntime, nx, ny, nz):
    ntile = 6

    def sample_iterator():
        # creates a timeseries where each time is the negation of time before it
        for _ in range(nsamples):
            start = {
                "a": np.random.uniform(
                    low=-1, high=1, size=(nbatch, 1, ntile, nx, ny, nz)
                ),
                "b": np.random.uniform(low=-1, high=1, size=(nbatch, 1, ntile, nx, ny)),
            }
            out = {key: [value] for key, value in start.items()}
            for _ in range(ntime - 1):
                for varname in start.keys():
                    out[varname].append(out[varname][-1] * -1.0)
            for varname in out:
                out[varname] = np.concatenate(out[varname], axis=1)
            yield out

    return iterable_to_tfdataset(list(sample_iterator()))


def tfdataset_to_xr_dataset(tfdataset, dims: Sequence[str]):
    """
    Returns a [time, tile, x, y, z] dataset needed for evaluation.

    Assumes input samples have shape [sample, time, tile, x, y(, z)], will
    concatenate samples along the time axis before returning.
    """
    data_sequences = collections.defaultdict(list)
    for sample in tfdataset:
        for name, value in sample.items():
            data_sequences[name].append(
                value.numpy().reshape(
                    [value.shape[0] * value.shape[1]] + list(value.shape[2:])
                )
            )
    data_vars = {}
    for name in data_sequences:
        data = np.concatenate(data_sequences[name])
        data_vars[name] = xr.DataArray(data, dims=dims[: len(data.shape)])
    return xr.Dataset(data_vars)


@pytest.mark.slow
@pytest.mark.parametrize("network_type", ["UNet"])
def test_train_graph_network(tmpdir, network_type):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    sizes = {"nbatch": 2, "ntime": 2, "nx": 6, "ny": 6, "nz": 2}
    state_variables = ["a", "b"]
    train_tfdataset = get_tfdataset(nsamples=20, **sizes)
    val_tfdataset = get_tfdataset(nsamples=3, **sizes)
    # for test, need one continuous series so we consistently flip sign
    test_sizes = {"nbatch": 1, "ntime": 100, "nx": 6, "ny": 6, "nz": 2}
    test_xrdataset = tfdataset_to_xr_dataset(
        get_tfdataset(nsamples=1, **test_sizes), dims=["time", "tile", "x", "y", "z"]
    )
    if network_type == "MPG":
        graph_network = MPGraphUNetConfig(
            num_step_message_passing=3, edge_hidden_features=4
        )
        training_config = AutoregressiveTrainingConfig(n_epoch=100)
        optimizer = OptimizerConfig(kwargs={"lr": 0.001})
    elif network_type == "UNet":
        graph_network = GraphUNetConfig()
        training_config = AutoregressiveTrainingConfig(n_epoch=30)
        optimizer = OptimizerConfig(kwargs={"lr": 0.005})

    hyperparameters = GraphHyperparameters(
        state_variables=state_variables,
        training_loop=training_config,
        optimizer_config=optimizer,
        graph_network=graph_network,
    )
    predictor = train_graph_model(hyperparameters, train_tfdataset, val_tfdataset)
    predicted, reference = predictor.predict(test_xrdataset, timesteps=1)
    bias = predicted.isel(time=1) - reference.isel(time=1)
    mean_bias: xr.Dataset = bias.mean()
    rmse: xr.Dataset = (bias ** 2).mean() ** 0.5
    for varname in state_variables:
        assert np.abs(mean_bias[varname]) < 0.1
        assert rmse[varname] < 0.1
