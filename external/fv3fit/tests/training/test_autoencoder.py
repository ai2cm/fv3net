import numpy as np
import xarray as xr
from typing import Sequence
from fv3fit.pytorch.cyclegan import AutoencoderHyperparameters, train_autoencoder
from fv3fit.pytorch.cyclegan.train import TrainingConfig
from fv3fit.tfdataset import iterable_to_tfdataset
import collections
import os
import fv3fit.pytorch
import fv3fit


def get_tfdataset(nsamples, nbatch, ntime, nx, ny, nz):
    ntile = 6

    grid_x = np.arange(0, nx, dtype=np.float32)
    grid_y = np.arange(0, ny, dtype=np.float32)
    grid_x, grid_y = np.broadcast_arrays(grid_x[:, None], grid_y[None, :])
    grid_x = grid_x[None, None, None, :, :, None]
    grid_y = grid_y[None, None, None, :, :, None]

    def sample_iterator():
        # creates a timeseries where each time is the negation of time before it
        for _ in range(nsamples):
            ax = np.random.uniform(0.1, 1.5, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            bx = np.random.uniform(8, 16, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            cx = np.random.uniform(0.0, 2 * np.pi, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            ay = np.random.uniform(0.1, 1.5, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            by = np.random.uniform(8, 16, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            cy = np.random.uniform(0.0, 2 * np.pi, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            a = (
                ax
                * np.sin(2 * np.pi * grid_x / bx + cx)
                * ay
                * np.sin(2 * np.pi * grid_y / by + cy)
            )
            start = {
                "a": a.astype(np.float32),
                "b": -a[..., 0].astype(np.float32),
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


def test_autoencoder(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx, ny = 32, 32
    sizes = {"nbatch": 2, "ntime": 2, "nx": nx, "ny": ny, "nz": 2}
    state_variables = ["a", "b"]
    train_tfdataset = get_tfdataset(nsamples=20, **sizes)
    val_tfdataset = get_tfdataset(nsamples=3, **sizes)
    hyperparameters = AutoencoderHyperparameters(
        state_variables=state_variables,
        generator=fv3fit.pytorch.GeneratorConfig(
            n_convolutions=2, n_resnet=3, max_filters=32
        ),
        training_loop=TrainingConfig(n_epoch=5, samples_per_batch=2),
        optimizer_config=fv3fit.pytorch.OptimizerConfig(name="Adam",),
        noise_amount=0.5,
    )
    predictor = train_autoencoder(hyperparameters, train_tfdataset, val_tfdataset)
    # for test, need one continuous series so we consistently flip sign
    test_sizes = {"nbatch": 1, "ntime": 100, "nx": nx, "ny": ny, "nz": 2}
    test_xrdataset = tfdataset_to_xr_dataset(
        get_tfdataset(nsamples=1, **test_sizes), dims=["time", "tile", "x", "y", "z"]
    )
    predicted = predictor.predict(test_xrdataset)
    reference = test_xrdataset
    # plotting code to uncomment if you'd like to manually check the results:
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
    mse: xr.Dataset = (bias ** 2).mean() ** 0.5
    for varname in state_variables:
        assert np.abs(mean_bias[varname]) < 0.1
        assert mse[varname] < 0.1


def test_autoencoder_overfit(tmpdir):
    fv3fit.set_random_seed(0)
    # run the test in a temporary directory to delete artifacts when done
    os.chdir(tmpdir)
    # need a larger nx, ny for the sample data here since we're training
    # on whether we can autoencode sin waves, and need to resolve full cycles
    nx, ny = 32, 32
    sizes = {"nbatch": 1, "ntime": 1, "nx": nx, "ny": ny, "nz": 2}
    state_variables = ["a", "b"]
    train_tfdataset = get_tfdataset(nsamples=1, **sizes)
    train_tfdataset = train_tfdataset.cache()  # needed to keep sample identical
    hyperparameters = AutoencoderHyperparameters(
        state_variables=state_variables,
        generator=fv3fit.pytorch.GeneratorConfig(
            n_convolutions=2, n_resnet=1, max_filters=32
        ),
        training_loop=TrainingConfig(n_epoch=100, samples_per_batch=6),
        optimizer_config=fv3fit.pytorch.OptimizerConfig(name="Adam",),
        noise_amount=0.0,
    )
    predictor = train_autoencoder(
        hyperparameters, train_tfdataset, validation_batches=None
    )
    # for test, need one continuous series so we consistently flip sign
    test_xrdataset = tfdataset_to_xr_dataset(
        train_tfdataset, dims=["time", "tile", "x", "y", "z"]
    )
    predicted = predictor.predict(test_xrdataset)
    reference = test_xrdataset
    # plotting code to uncomment if you'd like to manually check the results:
    # for i in range(6):
    #     fig, ax = plt.subplots(1, 2)
    #     vmin = reference["a"][0, i, :, :, 0].values.min()
    #     vmax = reference["a"][0, i, :, :, 0].values.max()
    #     ax[0].imshow(reference["a"][0, i, :, :, 0].values)  # , vmin=vmin, vmax=vmax)
    #     ax[1].imshow(predicted["a"][0, i, :, :, 0].values)  # , vmin=vmin, vmax=vmax)
    #     plt.tight_layout()
    #     plt.show()
    bias = predicted - reference
    mean_bias: xr.Dataset = bias.mean()
    rmse: xr.Dataset = (bias ** 2).mean()
    for varname in state_variables:
        assert np.abs(mean_bias[varname]) < 0.1
        assert rmse[varname] < 0.1
