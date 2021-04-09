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

    def load_metrics(self):
        return load_metrics(self._get_fs(), self.url)

    def load_diagnostics(self) -> Tuple[Metadata, Diagnostics]:
        return load_diagnostics(self._get_fs(), self.url)

    def find_movie_links(self):
        return find_movie_links(self._get_fs(), self.url)


@dataclass
class RunDiagnostics:
    """A collection of diagnostics from different runs, not all of which have
    the same variables
    
    """

    diagnostics: Diagnostics

    @property
    def runs(self) -> Sequence[str]:
        """The available runs"""
        return [d.run for d in self.diagnostics]

    @property
    def variables(self) -> Set[str]:
        """The available variables"""
        return set.union(*[set(d) for d in self.diagnostics])

    def get_variable(self, run: str, varname: Hashable) -> xr.DataArray:
        """Query a collection of diagnostics for a given run and variable

        Args:
            diagnostics: list of xarray datasets, each with a "run" attribute
            varname: variable to exctract from the expected run

        Returns:
            varname of run if present, otherwise nans with the expected
            metadata

        """
        diagnostics_dict = {d.run: d[varname] for d in self.diagnostics if varname in d}
        try:
            return diagnostics_dict[run]
        except KeyError:
            template = next(iter(diagnostics_dict.values()))
            return xr.full_like(template, np.nan)

    def is_baseline(self, run: str) -> bool:
        return [d.attrs["baseline"] for d in self.diagnostics if d.run == run][0]


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
    }
    for mean_filter, bias_filter in mean_bias_pairs.items():
        mean_vars = [var for var in ds if mean_filter in var]
        for var in mean_vars:
            matching_bias_var = var.replace(mean_filter, bias_filter)
            if matching_bias_var in ds:
                # verification = prognostic - bias
                verif_diagnostics[var] = ds[var] - ds[matching_bias_var]
                verif_diagnostics[var].attrs = ds[var].attrs
    return xr.Dataset(verif_diagnostics, attrs=verif_attrs)


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
