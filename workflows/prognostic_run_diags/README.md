## Prognostic run reports

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


### Running via argo

An argo workflow-template is provided that can easily be used to create a combined report. See `workflows/argo/README.md`.

### Developing

#### Adding/plotting a new diagnostic/metric

This system is designed for extension without extensively modifying the
source code. To accomplish this, it divides report generation into three
phases

1. diagnostic computation using `save_prognostic_run_diags.py`. A "diagnostic" is a multidimensional quantity 
   that is reasonably limited in size (e.g. can be easily downloaded).
1. Scalar "metrics" are computed from the output of the step above using `metrics.py`
1. Multiple sets of diagnostics/metrics are visualized in the static html report 
   generation script `generate_report.py`

#### Adding a diagnostic

A new diagnostics can be easily added by writing a new function in
`save_prognostic_run_diags.py` and decorating it with `add_to_diags`, see the
docs of that decorator for more details.

#### Adding a new metric

Similarly, a new metric can be added using the `metric` decorator in `metrics.py`. See that script for some examples.

#### Adding a new visualization

The steps above simply add a metric or diagnostic to the saved output, but do
not change the report.

To plot a new metric or diagnostic, add a function to `generate_report.py`
and decorate it with the `metrics_plot_manager.register` if its metric or
`diag_plot_manager.register` if its a diagnostic. This function needs to
return an object that is valid input to the `sections` argument of
report.create_report`, currently just a path to an image, or an object of
type `report.Plot`.

#### Testing

An integration test of the various steps required to generate a prognostic run report 
can be launched by calling `make test_prognostic_run_report` from the root of the 
fv3net repository.