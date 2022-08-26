from .base import TFDatasetLoader, register_tfdataset_loader
import dataclasses
from typing import Optional, Sequence, List
import tensorflow as tf
import numpy as np
from ..tfdataset import iterable_to_tfdataset
import dacite


@register_tfdataset_loader
@dataclasses.dataclass
class SyntheticNoise(TFDatasetLoader):
    nsamples: int
    nbatch: int
    ntime: int
    nx: int
    nz: int
    scalar_names: List[str] = dataclasses.field(default_factory=list)
    noise_amplitude: float = 1.0

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        dataset = get_noise_tfdataset(
            variable_names,
            scalar_names=self.scalar_names,
            nsamples=self.nsamples,
            nbatch=self.nbatch,
            ntime=self.ntime,
            nx=self.nx,
            ny=self.nx,
            nz=self.nz,
            noise_amplitude=self.noise_amplitude,
        )
        if local_download_path is not None:
            dataset = dataset.cache(local_download_path)
        return dataset

    @classmethod
    def from_dict(cls, d: dict) -> "TFDatasetLoader":
        return dacite.from_dict(
            data_class=cls, data=d, config=dacite.Config(strict=True)
        )


@register_tfdataset_loader
@dataclasses.dataclass
class SyntheticWaves(TFDatasetLoader):
    """
    Attributes:
        nsamples: number of samples to generate per batch
        nbatch: number of batches to generate
        nx: length of x- and y-dimensions to generate
        nz: length of z-dimension to generate
        scalar_names: names to generate as scalars instead of
            vertically-resolved variables
        scale_min: minimum amplitude of waves
        scale_max: maximum amplitude of waves
        period_min: minimum period of waves
        period_max: maximum period of waves
        phase_range: fraction of 2*pi to use for possible range of
            random phase, should be a value between 0 and 1.

    """

    nsamples: int
    nbatch: int
    ntime: int
    nx: int
    nz: int
    scalar_names: List[str] = dataclasses.field(default_factory=list)
    scale_min: float = 0.0
    scale_max: float = 1.0
    period_min: float = 8.0
    period_max: float = 16.0
    phase_range: float = 1.0

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        dataset = get_waves_tfdataset(
            variable_names,
            scalar_names=self.scalar_names,
            nsamples=self.nsamples,
            nbatch=self.nbatch,
            ntime=self.ntime,
            nx=self.nx,
            ny=self.nx,
            nz=self.nz,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            period_min=self.period_min,
            period_max=self.period_max,
            phase_range=self.phase_range,
        )
        if local_download_path is not None:
            dataset = dataset.cache(local_download_path)
        return dataset

    @classmethod
    def from_dict(cls, d: dict) -> "TFDatasetLoader":
        return dacite.from_dict(
            data_class=cls, data=d, config=dacite.Config(strict=True)
        )


def get_waves_tfdataset(
    variable_names,
    *,
    scalar_names,
    nsamples: int,
    nbatch: int,
    ntime: int,
    nx: int,
    ny: int,
    nz: int,
    scale_min: float,
    scale_max: float,
    period_min: float,
    period_max: float,
    phase_range: float,
):
    ntile = 6

    grid_x = np.arange(0, nx, dtype=np.float32)
    grid_y = np.arange(0, ny, dtype=np.float32)
    grid_x, grid_y = np.broadcast_arrays(grid_x[:, None], grid_y[None, :])
    grid_x = grid_x[None, None, None, :, :, None]
    grid_y = grid_y[None, None, None, :, :, None]

    def sample_iterator():
        # creates a timeseries where each time is the negation of time before it
        for _ in range(nsamples):
            ax = np.random.uniform(scale_min, scale_max, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            bx = np.random.uniform(period_min, period_max, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            cx = np.random.uniform(
                0.0, 2 * np.pi * phase_range, size=(nbatch, 1, ntile, nz)
            )[:, :, :, None, None, :]
            ay = np.random.uniform(scale_min, scale_max, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            by = np.random.uniform(period_min, period_max, size=(nbatch, 1, ntile, nz))[
                :, :, :, None, None, :
            ]
            cy = np.random.uniform(
                0.0, 2 * np.pi * phase_range, size=(nbatch, 1, ntile, nz)
            )[:, :, :, None, None, :]
            data = (
                ax
                * np.sin(2 * np.pi * grid_x / bx + cx)
                * ay
                * np.sin(2 * np.pi * grid_y / by + cy)
            )
            start = {}
            for varname in variable_names:
                if varname in scalar_names:
                    start[varname] = data[..., 0].astype(np.float32)
                else:
                    start[varname] = data.astype(np.float32)
            out = {key: [value] for key, value in start.items()}
            for _ in range(ntime - 1):
                for varname in start.keys():
                    out[varname].append(out[varname][-1] * -1.0)
            for varname in out:
                out[varname] = np.concatenate(out[varname], axis=1)
            yield out

    return iterable_to_tfdataset(list(sample_iterator()))


def get_noise_tfdataset(
    variable_names,
    *,
    scalar_names,
    nsamples: int,
    nbatch: int,
    ntime: int,
    nx: int,
    ny: int,
    nz: int,
    noise_amplitude: float,
):
    ntile = 6

    def sample_iterator():
        # creates a timeseries where each time is the negation of time before it
        for _ in range(nsamples):
            data = noise_amplitude * np.random.randn(nbatch, 1, ntile, nx, ny, nz)
            start = {}
            for varname in variable_names:
                if varname in scalar_names:
                    start[varname] = data[..., 0].astype(np.float32)
                else:
                    start[varname] = data.astype(np.float32)
            out = {key: [value] for key, value in start.items()}
            for _ in range(ntime - 1):
                for varname in start.keys():
                    out[varname].append(out[varname][-1] * -1.0)
            for varname in out:
                out[varname] = np.concatenate(out[varname], axis=1)
            yield out

    return iterable_to_tfdataset(list(sample_iterator()))
