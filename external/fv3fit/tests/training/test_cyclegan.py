import numpy as np
import xarray as xr
from typing import Sequence
from fv3fit.pytorch.cyclegan import (
    CycleGANHyperparameters,
    CycleGANNetworkConfig,
    CycleGANTrainingConfig,
    train_cyclegan,
)
from fv3fit.data import CycleGANLoader, SyntheticWaves, SyntheticNoise
import fv3fit.tfdataset
import tensorflow as tf
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
                period_min=8,
                period_max=16,
                type="sinusoidal",
            ),
            SyntheticWaves(
                nsamples=nsamples,
                nbatch=nbatch,
                ntime=ntime,
                nx=nx,
                nz=nz,
                scalar_names=["b"],
                scale_min=0.5,
                scale_max=1.0,
                period_min=8,
                period_max=16,
                type="square",
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


def test_cyclegan(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx = 32
    sizes = {"nbatch": 1, "ntime": 1, "nx": nx, "nz": 2}
    state_variables = ["a", "b"]
    train_tfdataset = get_tfdataset(nsamples=100, **sizes)
    val_tfdataset = get_tfdataset(nsamples=3, **sizes)
    hyperparameters = CycleGANHyperparameters(
        state_variables=state_variables,
        network=CycleGANNetworkConfig(
            generator=fv3fit.pytorch.GeneratorConfig(
                n_convolutions=2, n_resnet=5, max_filters=128, kernel_size=3
            ),
            generator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.001}
            ),
            discriminator=fv3fit.pytorch.DiscriminatorConfig(kernel_size=3),
            discriminator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.001}
            ),
            # identity_weight=0.01,
            # cycle_weight=0.3,
            # gan_weight=1.0,
            discriminator_weight=0.5,
        ),
        training_loop=CycleGANTrainingConfig(n_epoch=10, samples_per_batch=1),
    )
    predictor = train_cyclegan(hyperparameters, train_tfdataset, val_tfdataset)
    # for test, need one continuous series so we consistently flip sign
    real_a = tfdataset_to_xr_dataset(
        train_tfdataset.map(lambda a, b: a), dims=["time", "tile", "x", "y", "z"]
    )
    real_b = tfdataset_to_xr_dataset(
        train_tfdataset.map(lambda a, b: b), dims=["time", "tile", "x", "y", "z"]
    )
    output_a = predictor.predict(real_b, reverse=True)
    reconstructed_b = predictor.predict(output_a)
    # print("output a")
    # print_compare(output_a, real_a)
    # print("reconstructed b")
    # print_compare(reconstructed_b, real_b)
    output_b = predictor.predict(real_a)
    reconstructed_a = predictor.predict(output_b, reverse=True)
    # plotting code to uncomment if you'd like to manually check the results:
    iz = 0
    for i in range(1):
        fig, ax = plt.subplots(3, 2, figsize=(8, 8))
        vmin = -1.5
        vmax = 1.5
        ax[0, 0].imshow(real_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[0, 1].imshow(real_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[1, 0].imshow(output_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[1, 1].imshow(output_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[2, 0].imshow(
            reconstructed_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax
        )
        ax[2, 1].imshow(
            reconstructed_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax
        )
        ax[0, 0].set_title("real a")
        ax[0, 1].set_title("real b")
        ax[1, 0].set_title("output b")
        ax[1, 1].set_title("output a")
        ax[2, 0].set_title("reconstructed a")
        ax[2, 1].set_title("reconstructed b")
        plt.tight_layout()
        plt.show()
    # bias = predicted.isel(time=1) - reference.isel(time=1)
    # mean_bias: xr.Dataset = bias.mean()
    # mse: xr.Dataset = (bias ** 2).mean() ** 0.5
    # for varname in state_variables:
    #     assert np.abs(mean_bias[varname]) < 0.1
    #     assert mse[varname] < 0.1


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
                n_convolutions=2, n_resnet=1, max_filters=64
            ),
            generator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.001}
            ),
            discriminator_optimizer=fv3fit.pytorch.OptimizerConfig(
                name="Adam", kwargs={"lr": 0.0001}
            ),
            identity_weight=0.001,
            cycle_weight=0.3,
            gan_weight=1.0,
        ),
        training_loop=CycleGANTrainingConfig(n_epoch=200, samples_per_batch=6),
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
    output_a = predictor.predict(real_b, reverse=True)
    reconstructed_b = predictor.predict(output_a)
    print("output a")
    print_compare(output_a, real_a)
    print("reconstructed b")
    print_compare(reconstructed_b, real_b)
    output_b = predictor.predict(real_a)
    reconstructed_a = predictor.predict(output_b, reverse=True)
    print("reconstructed a")
    print_compare(reconstructed_a, real_a)
    print("output b")
    print_compare(output_b, real_b)
    # plotting code to uncomment if you'd like to manually check the results:
    # import pdb; pdb.set_trace()
    iz = 0
    for i in range(1):
        fig, ax = plt.subplots(3, 2, figsize=(12, 7))
        vmin = -1.5
        vmax = 1.5
        ax[0, 0].imshow(real_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[0, 1].imshow(real_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[1, 0].imshow(output_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[1, 1].imshow(output_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax)
        ax[2, 0].imshow(
            reconstructed_a["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax
        )
        ax[2, 1].imshow(
            reconstructed_b["a"][0, i, :, :, iz].values, vmin=vmin, vmax=vmax
        )
        ax[0, 0].set_title("real a")
        ax[0, 1].set_title("real b")
        ax[1, 0].set_title("output a")
        ax[1, 1].set_title("output b")
        ax[2, 0].set_title("reconstructed a")
        ax[2, 1].set_title("reconstructed b")
        plt.tight_layout()
        plt.show()
    # bias = predicted - reference
    # mean_bias: xr.Dataset = bias.mean()
    # rmse: xr.Dataset = (bias ** 2).mean()
    # for varname in state_variables:
    #     assert np.abs(mean_bias[varname]) < 0.1
    #     assert rmse[varname] < 0.1


def assert_close(a: xr.Dataset, b: xr.Dataset):
    rmse = ((a - b) ** 2).mean() ** 0.5
    bias = (a - b).mean()
    for varname in rmse.data_vars:
        assert rmse[varname] < 0.1
    for varname in bias.data_vars:
        assert bias[varname] < 0.1


def print_compare(a: xr.Dataset, b: xr.Dataset):
    rmse = ((a - b) ** 2).mean() ** 0.5
    bias = (a - b).mean()
    print("compare")
    for varname in rmse.data_vars:
        print(varname, rmse[varname], bias[varname])


def test_tuple_map():
    """
    External package test demonstrating that for map operations on tuples
    of functions, tuple entries are passed as independent arguments
    and must be collected with *args.
    """

    def generator():
        for entry in [(1, 1), (2, 2), (3, 3)]:
            yield entry

    dataset = tf.data.Dataset.from_generator(
        generator, output_types=(tf.int32, tf.int32)
    )

    def map_fn(x, y):
        return x * 2, y * 3

    mapped = dataset.map(map_fn)
    out = list(mapped)
    assert out == [(2, 3), (4, 6), (6, 9)]
