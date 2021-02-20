## Prognostic run reports

See Sphinx documentation TODO: ADD LINK.

### Metrics

| metric               | description                              |           |
|----------------------|------------------------------------------|-----------|
| rmse_3day/{variable} | average RMSE. (average of 3 hourly RMSE) | z500      |
| drift3day/{variable} | Day 3 average - Day 1 average            | tmplowest |

### Data Requirements

 The scripts require that provided prognostic runs have been post-processed
 so that their outputs are available in zarr format. See
 `fv3net/workflows/post_process_run/post_process.py`. Prognostic runs of any
 resolution that is a multiple of C48 can be handled by this report. However,
 it is possible that the grid for a particular resolution may need to be added
 to the catalog. See next section.


 ### Catalog entries

 Computing diagnostics requires certain entries in an intake catalog. By default,
 the catalog in the root of the fv3net repository is used. The catalog is assumed to
 contain the entries `grid/c48`, `grid/c96`, `40day_c48_atmos_8xdaily_may2020` and
 `40day_c48_gfsphysics_15min_may2020`.
