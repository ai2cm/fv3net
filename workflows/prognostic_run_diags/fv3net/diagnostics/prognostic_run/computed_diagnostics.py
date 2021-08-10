"""Utilities for loading computed diagnostics

"""
import json
from typing import Iterable, Hashable, Sequence, Tuple, Any, Set, Mapping
import os
import xarray as xr
import numpy as np
import fsspec
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import tempfile

from .metrics import metrics_registry
from .derived_diagnostics import derived_registry


__all__ = ["ComputedDiagnosticsList", "RunDiagnostics"]


GRID_VARS = ["area", "lonb", "latb", "lon", "lat", "land_sea_mask"]

Diagnostics = Iterable[xr.Dataset]
Metadata = Any


@dataclass
class ComputedDiagnosticsList:
    folders: Mapping[str, "DiagnosticFolder"]

    @staticmethod
    def from_directory(url: str) -> "ComputedDiagnosticsList":
        """Open a directory of computed diagnostics

        Args:
            url: URL to directory containing rundirs as subdirectories.
                "rundirs". rundirs are subdirectories of this bucket. They each
                contain diags.nc, metrics.json, and .mp4 files.
        """
        fs, _, _ = fsspec.get_fs_token_paths(url)
        return ComputedDiagnosticsList(detect_folders(url, fs))

    @staticmethod
    def from_urls(urls: Sequence[str]) -> "ComputedDiagnosticsList":
        """Open computed diagnostics at the specified urls
        """

        def url_to_folder(url):
            fs, _, path = fsspec.get_fs_token_paths(url)
            return DiagnosticFolder(fs, path[0])

        return ComputedDiagnosticsList(
            {str(k): url_to_folder(url) for k, url in enumerate(urls)}
        )

    @staticmethod
    def from_json(
        url: str, urls_are_rundirs: bool = False
    ) -> "ComputedDiagnosticsList":
        """Open labeled computed diagnostics at urls specified in given JSON."""

        def url_to_folder(url):
            fs, _, path = fsspec.get_fs_token_paths(url)
            return DiagnosticFolder(fs, path[0])

        with fsspec.open(url) as f:
            rundirs = json.load(f)

        if urls_are_rundirs:
            for item in rundirs:
                item["url"] += "_diagnostics"

        return ComputedDiagnosticsList(
            {item["name"]: url_to_folder(item["url"]) for item in rundirs}
        )

    def load_metrics(self) -> "RunMetrics":
        return RunMetrics(load_metrics(self.folders))

    def load_diagnostics(self) -> Tuple[Metadata, "RunDiagnostics"]:
        metadata, xarray_diags = load_diagnostics(self.folders)
        return metadata, RunDiagnostics(xarray_diags)

    def load_metrics_from_diagnostics(self) -> "RunMetrics":
        """Compute metrics on the fly from the pre-computed diagnostics."""
        return RunMetrics(load_metrics_from_diagnostics(self.folders))

    def find_movie_urls(self) -> "RunMovieUrls":
        movies = {name: folder.movie_urls for name, folder in self.folders.items()}
        return RunMovieUrls(movies)


@dataclass
class RunDiagnostics:
    """A collection of diagnostics from different runs, not all of which have
    the same variables
    
    """

    diagnostics: Diagnostics

    def __post_init__(self):
        # indexes for faster lookup
        self._attrs = {ds.run: ds.attrs for ds in self.diagnostics}
        self._varnames = {ds.run: set(ds) for ds in self.diagnostics}
        self._run_index = {ds.run: k for k, ds in enumerate(self.diagnostics)}

    @property
    def runs(self) -> Sequence[str]:
        """The available runs"""
        return list(self._run_index)

    @property
    def variables(self) -> Set[str]:
        """The available variables"""
        return set.union(*[set(d) for d in self.diagnostics])

    def _get_run(self, run: str) -> xr.Dataset:
        return self.diagnostics[self._run_index[run]]

    def get_variable(self, run: str, varname: Hashable) -> xr.DataArray:
        """Query a collection of diagnostics for a given run and variable

        Args:
            diagnostics: list of xarray datasets, each with a "run" attribute
            varname: variable to exctract from the expected run

        Returns:
            varname of run if present, otherwise nans with the expected
            metadata

        """
        if varname in self._varnames[run]:
            return self._get_run(run)[varname]
        else:
            for run in self._varnames:
                if varname in self._varnames[run]:
                    template = self._get_run(run)[varname]
                    return xr.full_like(template, np.nan)
            raise ValueError(f"{varname} not found.")

    def get_variables(self, run: str, varnames: Sequence[Hashable]) -> xr.Dataset:
        """Query a collection of diagnostics and return dataset of variables."""
        variables = [self.get_variable(run, v) for v in varnames]
        return xr.merge(variables)

    def matching_variables(self, varfilter: str) -> Set[str]:
        """The available variabes that include varfilter in their names."""
        return set(v for v in self.variables if varfilter in v)

    def is_baseline(self, run: str) -> bool:
        return self._attrs[run]["baseline"]


@dataclass
class RunMetrics:
    """A collection of metrics from different runs, not all of which have the
    same metrics"""

    metrics: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.metrics.empty

    @property
    def runs(self) -> Sequence[str]:
        """The available runs"""
        return list(self.metrics.run.unique())

    @property
    def types(self) -> Set[str]:
        """The available types of metrics"""
        metric_names = [self._prefix(m) for m in self.metrics.metric]
        return set(metric_names)

    def get_metric_variables(self, metric_type: str) -> Set[str]:
        """The available variables for given metric_type"""
        metric_names = [
            m for m in self.metrics.metric if self._prefix(m) == metric_type
        ]
        return set([self._suffix(m) for m in metric_names])

    def get_metric_value(self, metric_type: str, variable: str, run: str) -> float:
        m = self._get_metric(metric_type, variable, run)
        if m.empty:
            return np.nan
        else:
            return m.value.item()

    def get_metric_units(self, metric_type: str, variable: str, run: str) -> str:
        m = self._get_metric(metric_type, variable, run)
        if m.empty:
            return ""
        else:
            return m.units.item()

    def get_metric_all_runs(self, metric_type: str, variable: str) -> pd.Series:
        metric_name = self.metric_name(metric_type, variable)
        return self.metrics[self.metrics.metric == metric_name]

    @staticmethod
    def _prefix(metric: str) -> str:
        return metric.split("/")[0]

    @staticmethod
    def _suffix(metric: str) -> str:
        return metric.split("/")[1]

    @staticmethod
    def metric_name(metric_type: str, variable: str) -> str:
        return f"{metric_type}/{variable}"

    def _get_metric(self, metric_type: str, variable: str, run: str) -> pd.Series:
        _metrics = self.get_metric_all_runs(metric_type, variable)
        return _metrics[_metrics.run == run]


@dataclass
class RunMovieUrls:
    """Represents locations of movies for a collection of run diagnostics."""

    movies: Mapping[str, Sequence[str]]  # mapping from run name to sequence of URLs

    def by_movie_name(self):
        """Return mapping from movie name to sequence of (url, run_name) tuples."""
        movies_by_name = {}
        for run_name, urls in self.movies.items():
            for url in urls:
                movie_name = os.path.basename(url)
                movies_by_name.setdefault(movie_name, []).append((url, run_name))
        return movies_by_name


def load_metrics(rundirs) -> pd.DataFrame:
    """Load the metrics from a bucket"""
    metrics = _load_metrics(rundirs)
    return _metrics_dataframe_from_dict(metrics)


def load_metrics_from_diagnostics(rundirs) -> pd.DataFrame:
    """Load the diagnostics from a bucket and compute metrics"""
    metrics = {}
    _, diagnostics = load_diagnostics(rundirs)
    for ds in diagnostics:
        metrics[ds.run] = metrics_registry.compute(ds, n_jobs=1)
    return _metrics_dataframe_from_dict(metrics)


def _metrics_dataframe_from_dict(metrics) -> pd.DataFrame:
    metric_table = pd.DataFrame.from_records(_yield_metric_rows(metrics))
    run_table = parse_rundirs(list(metrics.keys()))
    return pd.merge(run_table, metric_table, on="run")


def load_diagnostics(rundirs) -> Tuple[Metadata, Diagnostics]:
    """Load metadata and merged diagnostics from a bucket"""
    diags = _load_diags(rundirs)
    run_table_lookup = parse_rundirs(rundirs)
    diagnostics = [
        ds.assign_attrs(run=key, **run_table_lookup.loc[key])
        for key, ds in diags.items()
    ]
    diagnostics = [convert_index_to_datetime(ds, "time") for ds in diagnostics]
    diagnostics = [_add_derived_diagnostics(ds) for ds in diagnostics]
    longest_run_ds = _longest_run(diagnostics)
    diagnostics.append(_get_verification_diagnostics(longest_run_ds))
    return get_metadata(diags), diagnostics


def _add_derived_diagnostics(ds):
    merged = xr.merge([ds, derived_registry.compute(ds, n_jobs=1)])
    return merged.assign_attrs(ds.attrs)


def _longest_run(diagnostics: Iterable[xr.Dataset]) -> xr.Dataset:
    max_length = 0
    for ds in diagnostics:
        if ds.sizes["time"] > max_length:
            longest_ds = ds
            max_length = ds.sizes["time"]
    return longest_ds


@dataclass
class DiagnosticFolder:
    """Represents the output of compute diagnostics"""

    fs: fsspec.AbstractFileSystem
    path: str

    @property
    def metrics(self):
        path = os.path.join(self.path, "metrics.json")
        return json.loads(self.fs.cat(path))

    @property
    def diagnostics(self) -> xr.Dataset:
        path = os.path.join(self.path, "diags.nc")
        with tempfile.NamedTemporaryFile() as f:
            self.fs.get(path, f.name)
            return xr.open_dataset(f.name, engine="h5netcdf").compute()

    @property
    def movie_urls(self) -> Sequence[str]:
        movie_paths = self.fs.glob(os.path.join(self.path, "*.mp4"))
        if "gs" in self.fs.protocol:
            movie_paths = ["gs://" + path for path in movie_paths]
        return movie_paths


def detect_folders(
    bucket: str, fs: fsspec.AbstractFileSystem,
) -> Mapping[str, DiagnosticFolder]:
    diag_ncs = fs.glob(os.path.join(bucket, "*", "diags.nc"))
    return {
        Path(url).parent.name: DiagnosticFolder(fs, Path(url).parent.as_posix())
        for url in diag_ncs
    }


def _load_diags(rundirs: Mapping[str, DiagnosticFolder]):
    metrics = {}
    for rundir, diag_folder in rundirs.items():
        metrics[rundir] = diag_folder.diagnostics
    return metrics


def _yield_metric_rows(metrics):
    """yield rows to be combined into a dataframe
    """
    for run in metrics:
        for name in metrics[run]:
            yield {
                "run": run,
                "metric": name,
                "value": metrics[run][name]["value"],
                "units": metrics[run][name]["units"],
            }


def _load_metrics(rundirs):
    metrics = {}
    for rundir, diag_folder in rundirs.items():
        metrics[rundir] = diag_folder.metrics
    return metrics


def parse_rundirs(rundirs) -> pd.DataFrame:
    run_table = pd.DataFrame.from_records(_parse_metadata(run) for run in rundirs)
    return run_table.set_index("run")


def _parse_metadata(run: str):
    baseline_s = "-baseline"

    if run.endswith(baseline_s):
        baseline = True
    else:
        baseline = False

    return {"run": run, "baseline": baseline}


def _get_verification_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """Back out verification diagnostics from prognostic run values and biases"""
    verif_diagnostics = {}
    verif_attrs = {"run": "verification", "baseline": True}
    mean_bias_pairs = {
        "spatial_mean": "mean_bias",
        "diurn_component": "diurn_bias",
        "zonal_and_time_mean": "zonal_bias",
        "zonal_mean_value": "zonal_mean_bias",
        "time_mean_value": "time_mean_bias",
        "histogram": "hist_bias",
        "pressure_level_zonal_time_mean": "pressure_level_zonal_bias",
    }
    for mean_filter, bias_filter in mean_bias_pairs.items():
        mean_vars = [var for var in ds if mean_filter in var]
        for var in mean_vars:
            matching_bias_var = var.replace(mean_filter, bias_filter)
            if matching_bias_var in ds:
                # verification = prognostic - bias
                verif_diagnostics[var] = ds[var] - ds[matching_bias_var]
                verif_diagnostics[var].attrs = ds[var].attrs
    # special handling for histogram bin widths
    bin_width_vars = [var for var in ds if "bin_width_histogram" in var]
    for var in bin_width_vars:
        verif_diagnostics[var] = ds[var]
    verif_dataset = xr.Dataset(verif_diagnostics)
    return xr.merge([ds[GRID_VARS], verif_dataset]).assign_attrs(verif_attrs)


def get_metadata(diags):
    run_urls = {key: ds.attrs["url"] for key, ds in diags.items()}
    verification_datasets = [ds.attrs["verification"] for ds in diags.values()]
    if any([verification_datasets[0] != item for item in verification_datasets]):
        raise ValueError(
            "Report cannot be generated with diagnostics computed against "
            "different verification datasets."
        )
    verification_label = {"verification dataset": verification_datasets[0]}
    return {**verification_label, **run_urls}


def convert_index_to_datetime(ds, dim):
    return ds.assign_coords({dim: ds.indexes[dim].to_datetimeindex()})
