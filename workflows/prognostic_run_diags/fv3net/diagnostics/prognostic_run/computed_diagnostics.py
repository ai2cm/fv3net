"""Utilities for loading computed diagnostics

"""
import json
from typing import Iterable, Hashable, Sequence, Tuple, Any, Set
import os
import xarray as xr
import numpy as np
import fsspec
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


__all__ = ["ComputedDiagnosticsList", "RunDiagnostics"]


PUBLIC_GCS_DOMAIN = "https://storage.googleapis.com"
GRID_VARS = ["area", "lonb", "latb", "lon", "lat"]

Diagnostics = Iterable[xr.Dataset]
Metadata = Any


@dataclass
class ComputedDiagnosticsList:
    """Represents a list of computed diagnostics

    Attrs:
        url: URL to directory containing rundirs as subdirectories.
            "rundirs". rundirs are subdirectories of this bucket. They each
            contain diags.nc, metrics.json, and .mp4 files.
    """

    url: str

    def _get_fs(self):
        fs, _, _ = fsspec.get_fs_token_paths(self.url)
        return fs

    def load_metrics(self) -> "RunMetrics":
        return RunMetrics(load_metrics(self._get_fs(), self.url))

    def load_diagnostics(self) -> Tuple[Metadata, "RunDiagnostics"]:
        metadata, xarray_diags = load_diagnostics(self._get_fs(), self.url)
        return metadata, RunDiagnostics(xarray_diags)

    def find_movie_links(self):
        return find_movie_links(self._get_fs(), self.url)


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


def load_metrics(fs, bucket) -> pd.DataFrame:
    """Load the metrics from a bucket"""
    rundirs = detect_rundirs(bucket, fs)
    metrics = _load_metrics(bucket, rundirs)
    metric_table = pd.DataFrame.from_records(_yield_metric_rows(metrics))
    run_table = parse_rundirs(rundirs)
    return pd.merge(run_table, metric_table, on="run")


def load_diagnostics(fs, bucket) -> Tuple[Metadata, Diagnostics]:
    """Load metadata and merged diagnostics from a bucket"""
    rundirs = detect_rundirs(bucket, fs)
    diags = _load_diags(bucket, rundirs)
    run_table_lookup = parse_rundirs(rundirs)
    diagnostics = [
        ds.assign_attrs(run=key, **run_table_lookup.loc[key])
        for key, ds in diags.items()
    ]
    diagnostics = [convert_index_to_datetime(ds, "time") for ds in diagnostics]

    # hack to add verification data from longest set of diagnostics as new run
    # TODO: generate separate diags.nc file for verification data and load that in here
    longest_run_ds = _longest_run(diagnostics)
    diagnostics.append(_get_verification_diagnostics(longest_run_ds))
    return get_metadata(diags), diagnostics


def find_movie_links(fs, bucket, domain=PUBLIC_GCS_DOMAIN):
    """Get the movie links from a bucket

    Returns:
        A dictionary of (public_url, rundir) tuples
    """
    rundirs = detect_rundirs(bucket, fs)

    # TODO refactor to split out I/O from html generation
    movie_links = {}
    for rundir in rundirs:
        movie_paths = fs.glob(os.path.join(bucket, rundir, "*.mp4"))
        for gcs_path in movie_paths:
            movie_name = os.path.basename(gcs_path)
            if movie_name not in movie_links:
                movie_links[movie_name] = []
            public_url = os.path.join(domain, gcs_path)
            movie_links[movie_name].append((public_url, rundir))
    return movie_links


def _longest_run(diagnostics: Iterable[xr.Dataset]) -> xr.Dataset:
    max_length = 0
    for ds in diagnostics:
        if ds.sizes["time"] > max_length:
            longest_ds = ds
            max_length = ds.sizes["time"]
    return longest_ds


def detect_rundirs(bucket: str, fs: fsspec.AbstractFileSystem):
    diag_ncs = fs.glob(os.path.join(bucket, "*", "diags.nc"))
    if len(diag_ncs) < 2:
        raise ValueError(
            "Plots require more than 1 diagnostic directory in"
            f" {bucket} for holoviews plots to display correctly."
        )
    return [Path(url).parent.name for url in diag_ncs]


def _load_diags(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "diags.nc")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = xr.open_dataset(f, engine="h5netcdf").compute()
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


def _load_metrics(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "metrics.json")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = json.load(f)

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
    """Back out verification timeseries from prognostic run value and bias"""
    verif_diagnostics = {}
    verif_attrs = {"run": "verification", "baseline": True}
    mean_bias_pairs = {
        "spatial_mean": "mean_bias",
        "diurn_component": "diurn_bias",
        "zonal_and_time_mean": "zonal_bias",
        "zonal_mean_value": "zonal_mean_bias",
        "time_mean_value": "time_mean_bias",
    }
    for mean_filter, bias_filter in mean_bias_pairs.items():
        mean_vars = [var for var in ds if mean_filter in var]
        for var in mean_vars:
            matching_bias_var = var.replace(mean_filter, bias_filter)
            if matching_bias_var in ds:
                # verification = prognostic - bias
                verif_diagnostics[var] = ds[var] - ds[matching_bias_var]
                verif_diagnostics[var].attrs = ds[var].attrs
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
