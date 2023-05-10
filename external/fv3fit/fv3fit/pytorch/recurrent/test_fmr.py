import numpy as np
import xarray as xr
from typing import Sequence
import pytest
from fv3fit.pytorch.recurrent import (
    FMRHyperparameters,
    FMRNetworkConfig,
    FMRTrainingConfig,
    train_fmr,
)
from fv3fit.data import SyntheticNoise
import fv3fit.tfdataset
import collections
import os
import fv3fit.pytorch
import fv3fit
import fv3fit.pytorch.recurrent.discriminator


def get_tfdataset(nsamples, nbatch, ntime, nx, nz):
    config = SyntheticNoise(
        nsamples=nsamples,
        nbatch=nbatch,
        ntime=ntime,
        nx=nx,
        nz=nz,
        scalar_names=["var_2d"],
        noise_amplitude=1.0,
    )
    dataset = config.open_tfdataset(
        local_download_path=None, variable_names=["var_3d", "var_2d"]
    )
    return dataset


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
def test_fmr_runs_without_errors(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx = 48
    sizes = {"nbatch": 1, "ntime": 4, "nx": nx, "nz": 2}
    state_variables = ["var_3d", "var_2d"]
    train_tfdataset = get_tfdataset(nsamples=3, **sizes)
    val_tfdataset = get_tfdataset(nsamples=2, **sizes)
    hyperparameters = FMRHyperparameters(
        state_variables=state_variables,
        network=FMRNetworkConfig(
            generator=fv3fit.pytorch.RecurrentGeneratorConfig(
                n_convolutions=2, n_resnet=5, max_filters=128, kernel_size=3
            ),
            generator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.001}
            ),
            discriminator=fv3fit.pytorch.recurrent.discriminator.DiscriminatorConfig(
                kernel_size=3
            ),
            discriminator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.001}
            ),
            generator_weight=1.0,
            discriminator_weight=0.5,
            target_weight=1.0,
        ),
        training=FMRTrainingConfig(
            n_epoch=2, samples_per_batch=2, validation_batch_size=2
        ),
    )
    with fv3fit.wandb.disable_wandb():
        predictor = train_fmr(hyperparameters, train_tfdataset, val_tfdataset)
    # for test, need one continuous series so we consistently flip sign
    real = tfdataset_to_xr_dataset(
        train_tfdataset, dims=["time", "tile", "x", "y", "z"]
    )
    predictor.predict(real, timesteps=3)
    # We can't use regtest because the output is not deterministic between platforms,
    # but you can un-comment this and use local-only (do not commit to git) regtest
    # outputs when refactoring the code to ensure you don't change results.
    # import json
    # import vcm.testing
    # regtest.write(json.dumps(vcm.testing.checksum_dataarray_mapping(output_a)))
    # regtest.write(json.dumps(vcm.testing.checksum_dataarray_mapping(reconstructed_b)))
    # regtest.write(json.dumps(vcm.testing.checksum_dataarray_mapping(output_b)))
    # regtest.write(json.dumps(vcm.testing.checksum_dataarray_mapping(reconstructed_a)))
