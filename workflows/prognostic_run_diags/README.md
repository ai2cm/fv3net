## Prognostic run reports

### Metrics

| metric |  description| |
|-|-|-|
|rmse_3day/{variable} | average RMSE. (average of 3 hourly RMSE)| z500 |
|drift3day/{variable} |  Day 3 average - Day 1 average | tmplowest |

### Usage

This folder contains a workflow for saving prognostic run metrics to netCDF
files and then uploading [this report][1] to a public bucket. 

It requires that argo be installed and the kubectl tool is properly configured.

To generate reports for all the directories in `rundirs.json` to the cluster,
simply run

    bash run_all.sh rundirs.json docker_image

where `docker_image` is full name of the specific fv3net image you wish to generate 
the report with. Using the latest master commit for the fv3net image is a good bet.

This command will make output like this:

    Name:                prognostic-run-diags-sps8h
    Namespace:           default
    ServiceAccount:      default
    Status:              Pending
    Created:             Thu Apr 30 12:01:17 -0700 (now)

The data are stored at a directory based on the "Name" above. Specifically, the computed outputs wil be located at `gs://vcm-ml-public/argo/{{workflow.name}}`. The published report will be available at the url:

    http://storage.googleapis.com/vcm-ml-public/argo/{{workflow.name}}/index.html

This job can be monitored by running

    argo watch <name of pod in last command>

### Dependencies

- md5sum. Usually available on linux. To install on mac run: `brew install md5sha1sum`.
- argo CLI
- yq

### Generating a new report

Simply add a new item to rundirs.yml and resubmit the job. All the steps will be
re-run, which is redundant, but the process isn't that slow. The workflow requires
that provided prognostic runs have been post-processed so that their outputs are
available in zarr format. See `fv3net/workflows/prognostic_c48_run/post_process.py`.


[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html

## Adding/plotting a new diagnostic/metric

This system is designed for extension without extensively modifying the
source code. To accomplish this, it divides report generation into three
phases

1. diagnostic computation using `save_prognostic_run_diags.py`. A "diagnostic" is a multidimensional quantity 
   that is reasonably limited in size (e.g. can be easily downloaded).
1. Scalar "metrics" are computed from the output of the step above using `metrics.py`
1. Multiple sets of diagnostics/metrics are visualized in the static html report 
   generation script `generate_report.py`

### Adding a diagnostic

A new diagnostics can be easily added by writing a new function in
`save_prognostic_run_diags.py` and decorating it with `add_to_diags`, see the
docs of that decorator for more details.

### Adding a new metric

Similarly, a new metric can be added using the `metric` decorator in `metrics.py`. See that script for some examples.

### Adding a new visualization

The steps above simply add a metric or diagnostic to the saved output, but do
not change the report.

To plot a new metric or diagnostic, add a function to `generate_report.py`
and decorate it with the `metrics_plot_manager.register` if its metric or
`diag_plot_manager.register` if its a diagnostic. This function needs to
return an object that is valid input to the `sections` argument of
report.create_report`, currently just a path to an image, or an object of
type `report.Plot`.
