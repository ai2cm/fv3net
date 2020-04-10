## Prognostic run reports

### Metrics

| metric |  description| |
|-|-|-|
|rmse_3day/{variable} | average RMSE. (average of 3 hourly RMSE)| z500 |
|drift3day/{variable} |  Day 3 average - Day 1 average | tmplowest |


### Structure

This folder contains a workflow for saving prognostic run metrics to netCDF
files and then uploading [this report][1] to a public bucket. Here is the structure
of this directory:

	├── README.md
	├── argo.yaml        # argo pipeline
	├── combined.ipynb   # ipython notebook for combined report
	├── index.ipynb      # ipython notebook for single run reports
	├── putfile.py       # script for uploading data to GCS (uses gcsfs)
	├── run_all.sh       # main entry point
	├── rundirs.yml      # list of prognostic runs and tags to generate diags for
	└── upload_report.sh # script for running/uploading the combined.ipynb notebook



This workflow depends on an up-to-date fv3net image. To generate one, run

    make push_image

from the root directory of fv3net. It also requires that argo be installed and the kubectl tool is properly configured.

To generate reports for all the directories in `rundirs.yml` to the cluster,
simply run

    bash run_all.sh

from this directory. This job can be monitored by running

    argo watch <name of pod in last command>

### Generating a new report

Simply add a new item to rundirs.yml and resubmit the job. All the steps will be
re-run, which is redundant, but the process isn't that slow.

[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html

### Updating the code

After updating the script `fv3net/pipelines/save_prognostic_run_diags.py` you will need to rebuild the docker image, and update the corresponding references in the `argo.yaml`. Specifically, `make push_image` will output a SHA256 digest for the pushed-docker image. Copy-paste this into the "image" fields in `argo.yaml`.
