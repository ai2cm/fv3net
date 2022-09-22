import numpy as np
import xarray as xr
from typing import Sequence
from fv3fit.pytorch.cyclegan import AutoencoderHyperparameters, train_autoencoder
from fv3fit.pytorch.cyclegan.train_autoencoder import TrainingConfig
import pytest
from fv3fit.data.synthetic import SyntheticWaves, SyntheticNoise
import collections
import os
import fv3fit.pytorch
import fv3fit


def get_synthetic_waves_tfdataset(nsamples, nbatch, ntime, nx, nz):
    """
    Returns a tfdataset of synthetic waves with varying period and amplitude.

    Samples are dictionaries of tensors with shape
    [batch, sample, time, tile, x, y, z].

    Dataset contains a variable "a" which is vertically-resolved
    and "b" which is a scalar.
    """
    config = SyntheticWaves(
        nsamples=nsamples,
        nbatch=nbatch,
        ntime=ntime,
        nx=nx,
        nz=nz,
        wave_type="sinusoidal",
        scalar_names=["b"],
        scale_min=1.0,
        scale_max=1.0,
        period_min=8,
        period_max=16,
    )
    dataset = config.open_tfdataset(local_download_path=None, variable_names=["a", "b"])
    return dataset


def get_noise_tfdataset(nsamples, nbatch, ntime, nx, nz):
    """
    Returns a tfdataset of random noise.

    Dataset contains a variable "a" which is vertically-resolved
    and "b" which is a scalar.
    """
    config = SyntheticNoise(
        nsamples=nsamples,
        nbatch=nbatch,
        ntime=ntime,
        nx=nx,
        nz=nz,
        noise_amplitude=1.0,
        scalar_names=["b"],
    )
    dataset = config.open_tfdataset(local_download_path=None, variable_names=["a", "b"])
    return dataset


def tfdataset_to_xr_dataset(tfdataset, dims: Sequence[str]):
    """
    Takes a tfdataset whose samples all have the same shape, and converts
    it to an xarray dataset with the given dimensions.

    Combines the first two dimensions into a single dimension labelled
    according to the first entry of `dims`. This is done because we need
    to convert [batch, sample] dimensions needed for tfdataset training
    into a single [time] dimension which matches the xarray datasets we see
    in production.
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
def test_autoencoder(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx = 32
    sizes = {"nbatch": 2, "ntime": 2, "nx": nx, "nz": 2}
    state_variables = ["a", "b"]
    # dataset is random sinusoidal waves with varying amplitude and period
    # doesn't particularly matter what the input data is, as long as the denoising
    # autoencoder can learn to remove noise from its samples. A dataset of
    # pure synthetic noise would not work, it must have some structure.
    train_tfdataset = get_synthetic_waves_tfdataset(nsamples=100, **sizes)
    val_tfdataset = get_synthetic_waves_tfdataset(nsamples=3, **sizes)
    hyperparameters = AutoencoderHyperparameters(
        state_variables=state_variables,
        generator=fv3fit.pytorch.GeneratorConfig(
            n_convolutions=1, n_resnet=3, max_filters=32
        ),
        training_loop=TrainingConfig(n_epoch=5, samples_per_batch=2),
        optimizer_config=fv3fit.pytorch.OptimizerConfig(name="Adam",),
        noise_amount=0.5,
    )
    predictor = train_autoencoder(hyperparameters, train_tfdataset, val_tfdataset)
    test_sizes = {"nbatch": 1, "ntime": 100, "nx": nx, "nz": 2}
    # predict takes xarray datasets, so we have to convert
    test_xrdataset = tfdataset_to_xr_dataset(
        get_synthetic_waves_tfdataset(nsamples=1, **test_sizes),
        dims=["time", "tile", "x", "y", "z"],
    )
    predicted = predictor.predict(test_xrdataset)
    reference = test_xrdataset
    # plotting code to uncomment if you'd like to manually check the results:
    # import matplotlib.pyplot as plt
    # for i in range(6):
    #     fig, ax = plt.subplots(1, 2)
    #     vmin = reference["a"][0, i, :, :, 0].values.min()
    #     vmax = reference["a"][0, i, :, :, 0].values.max()
    #     ax[0].imshow(reference["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
    #     ax[1].imshow(predicted["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
    #     plt.tight_layout()
    #     plt.show()
    bias = predicted.isel(time=1) - reference.isel(time=1)
    mean_bias: xr.Dataset = bias.mean()
    mse: xr.Dataset = (bias ** 2).mean()
    for varname in state_variables:
        assert np.abs(mean_bias[varname]) < 0.1
        assert mse[varname] < 0.1


@pytest.mark.slow
def test_autoencoder_overfit(tmpdir):
    """
    Test that the autoencoder training function can overfit on a single sample.

    This is an easier problem than fitting on a training dataset and testing
    on a validation dataset.
    """
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx = 16
    sizes = {"nbatch": 1, "ntime": 1, "nx": nx, "nz": 2}
    state_variables = ["a", "b"]
    # for single-sample overfitting we can use any data, even pure noise
    train_tfdataset = get_noise_tfdataset(nsamples=1, **sizes)
    train_tfdataset = train_tfdataset.cache()  # needed to keep sample identical
    hyperparameters = AutoencoderHyperparameters(
        state_variables=state_variables,
        generator=fv3fit.pytorch.GeneratorConfig(
            n_convolutions=1, n_resnet=1, max_filters=16
        ),
        training_loop=TrainingConfig(n_epoch=100, samples_per_batch=6),
        optimizer_config=fv3fit.pytorch.OptimizerConfig(name="Adam",),
        noise_amount=0.0,
    )
    predictor = train_autoencoder(
        hyperparameters, train_tfdataset, validation_batches=None
    )
    fv3fit.dump(predictor, str(tmpdir))
    predictor = fv3fit.load(str(tmpdir))
    # predict takes xarray datasets, so we have to convert
    test_xrdataset = tfdataset_to_xr_dataset(
        train_tfdataset, dims=["time", "tile", "x", "y", "z"]
    )

    predicted = predictor.predict(test_xrdataset)
    reference = test_xrdataset
    # plotting code to uncomment if you'd like to manually check the results:
    import matplotlib.pyplot as plt

    for i in range(6):
        fig, ax = plt.subplots(1, 2)
        vmin = reference["a"][0, i, :, :, 0].values.min()
        vmax = reference["a"][0, i, :, :, 0].values.max()
        ax[0].imshow(reference["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        ax[1].imshow(predicted["a"][0, i, :, :, 0].values, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.show()
    bias = predicted - reference
    mean_bias: xr.Dataset = bias.mean()
    mse: xr.Dataset = (bias ** 2).mean()
    for varname in state_variables:
        assert np.abs(mean_bias[varname]) < 0.1
        assert mse[varname] < 0.2
