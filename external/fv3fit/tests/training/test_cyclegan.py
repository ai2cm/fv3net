import numpy as np
import xarray as xr
from typing import Sequence
from fv3fit.pytorch.cyclegan import (
    CycleGANHyperparameters,
    CycleGANNetworkConfig,
    CycleGANTrainingConfig,
    train_cyclegan,
    CycleGAN,
)
from fv3fit.data import CycleGANLoader, SyntheticWaves, SyntheticNoise
import collections
import os
import fv3fit.pytorch
import fv3fit
import matplotlib.pyplot as plt


def get_tfdataset(nsamples, nbatch, ntime, nx, nz):
    config = CycleGANLoader(
        domain_configs=[
            SyntheticWaves(
                nsamples=nsamples,
                nbatch=nbatch,
                ntime=ntime,
                nx=nx,
                nz=nz,
                scalar_names=["b"],
                scale_min=0.5,
                scale_max=1.0,
                period_min=4,
                period_max=7,
                phase_range=0.1,
            ),
            SyntheticWaves(
                nsamples=nsamples,
                nbatch=nbatch,
                ntime=ntime,
                nx=nx,
                nz=nz,
                scalar_names=["b"],
                scale_min=1.0,
                scale_max=1.5,
                period_min=8,
                period_max=16,
                phase_range=0.1,
            ),
        ]
    )
    dataset = config.open_tfdataset(local_download_path=None, variable_names=["a", "b"])
    return dataset


def get_noise_tfdataset(nsamples, nbatch, ntime, nx, nz):
    config = CycleGANLoader(
        domain_configs=[
            SyntheticNoise(
                nsamples=nsamples,
                nbatch=nbatch,
                ntime=ntime,
                nx=nx,
                nz=nz,
                noise_amplitude=1.0,
            ),
            SyntheticNoise(
                nsamples=nsamples,
                nbatch=nbatch,
                ntime=ntime,
                nx=nx,
                nz=nz,
                noise_amplitude=1.0,
            ),
        ]
    )
    dataset = config.open_tfdataset(local_download_path=None, variable_names=["a", "b"])
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


# def test_cyclegan(tmpdir):
#     fv3fit.set_random_seed(0)
#     # run the test in a temporary directory to delete artifacts when done
#     os.chdir(tmpdir)
#     # need a larger nx, ny for the sample data here since we're training
#     # on whether we can autoencode sin waves, and need to resolve full cycles
#     nx, ny = 32, 32
#     sizes = {"nbatch": 2, "ntime": 2, "nx": nx, "ny": ny, "nz": 2}
#     state_variables = ["a", "b"]
#     train_tfdataset = get_tfdataset(nsamples=20, **sizes)
#     val_tfdataset = get_tfdataset(nsamples=3, **sizes)
#     hyperparameters = CycleGANHyperparameters(
#         state_variables=state_variables,
#         generator=fv3fit.pytorch.GeneratorConfig(
#             n_convolutions=2, n_resnet=3, max_filters=32
#         ),
#         training_loop=TrainingConfig(n_epoch=5, samples_per_batch=2),
#         optimizer_config=fv3fit.pytorch.OptimizerConfig(name="Adam",),
#         noise_amount=0.5,
#     )
#     predictor = train_cyclegan(hyperparameters, train_tfdataset, val_tfdataset)
#     # for test, need one continuous series so we consistently flip sign
#     test_sizes = {"nbatch": 1, "ntime": 100, "nx": nx, "ny": ny, "nz": 2}
#     test_xrdataset = tfdataset_to_xr_dataset(
#         get_tfdataset(nsamples=1, **test_sizes), dims=["time", "tile", "x", "y", "z"]
#     )
#     predicted = predictor.predict(test_xrdataset)
#     reference = test_xrdataset
#     # plotting code to uncomment if you'd like to manually check the results:
#     # for i in range(6):
#     #     fig, ax = plt.subplots(1, 2)
#     #     vmin = reference["a"][0, i, :, :, 0].values.min()
#     #     vmax = reference["a"][0, i, :, :, 0].values.max()
#     #     ax[0].imshow(reference["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
#     #     ax[1].imshow(predicted["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
#     #     plt.tight_layout()
#     #     plt.show()
#     bias = predicted.isel(time=1) - reference.isel(time=1)
#     mean_bias: xr.Dataset = bias.mean()
#     mse: xr.Dataset = (bias ** 2).mean() ** 0.5
#     for varname in state_variables:
#         assert np.abs(mean_bias[varname]) < 0.1
#         assert mse[varname] < 0.1


def test_cyclegan_overfit(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx = 16
    sizes = {"nbatch": 1, "ntime": 1, "nx": nx, "nz": 2}
    state_variables = ["a", "b"]
    train_tfdataset = get_noise_tfdataset(nsamples=1, **sizes)
    train_tfdataset = train_tfdataset.cache()  # needed to keep sample identical
    hyperparameters = CycleGANHyperparameters(
        state_variables=state_variables,
        network=CycleGANNetworkConfig(
            generator=fv3fit.pytorch.GeneratorConfig(
                n_convolutions=2, n_resnet=1, max_filters=128
            ),
            generator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.01}
            ),
            discriminator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.01}
            ),
        ),
        training_loop=CycleGANTrainingConfig(n_epoch=100, samples_per_batch=6),
    )
    predictor = train_cyclegan(
        hyperparameters, train_tfdataset, validation_batches=train_tfdataset
    )
    # for test, need one continuous series so we consistently flip sign
    real_a = tfdataset_to_xr_dataset(
        train_tfdataset.map(lambda a, b: a), dims=["time", "tile", "x", "y", "z"]
    )
    real_b = tfdataset_to_xr_dataset(
        train_tfdataset.map(lambda a, b: b), dims=["time", "tile", "x", "y", "z"]
    )
    output_b = predictor.predict(real_a)
    output_a = predictor.predict(real_b, reverse=True)
    # plotting code to uncomment if you'd like to manually check the results:
    for i in range(3):
        fig, ax = plt.subplots(2, 2)
        vmin = -1.5
        vmax = 1.5
        ax[0, 0].imshow(real_a["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        ax[0, 1].imshow(real_b["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        ax[1, 0].imshow(output_a["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        ax[1, 1].imshow(output_b["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        ax[0, 0].set_title("real a")
        ax[0, 1].set_title("real b")
        ax[1, 0].set_title("output a")
        ax[1, 1].set_title("output b")
        plt.tight_layout()
        plt.show()
    # bias = predicted - reference
    # mean_bias: xr.Dataset = bias.mean()
    # rmse: xr.Dataset = (bias ** 2).mean()
    # for varname in state_variables:
    #     assert np.abs(mean_bias[varname]) < 0.1
    #     assert rmse[varname] < 0.1
