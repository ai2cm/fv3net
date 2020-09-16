# Training and Prognostic run workflow

This directory contains an example workflow for running the end to end
workflow starting with pre-computed training data. It performs the following
operations:
- train
- offline report
- prognostic run
- compute prognostic run diagnostic netCDFs and json files. A report combining
  several runs should be run separately.

The configurations are contained in three places:

1. `prognostic-run.yaml` contains the prognostic run configurations
1. `training-config.yaml` contains the scikit learn settings
1. `run.sh` contains other configurations, such as the output location,
   timestep selection, fv3net commit SHA, etc.


## Recommended workflow

First make a branch tagged with the date:

    git co -b experiment/$(date +%F)/my-experiment


Then, edit the configuration files in place. To ease development, `run.sh`
currently points to shortend lists of training/testing times in these lines:

    -p train-times="$(<  ../train_short.json)" \
    -p test-times="$(<  ../test_short.json)" \


Before proceeding to full experiments, you may want to open a PR to
vcm-workflow-control.

Once your configurations are ready, submit a testing run by running `make`. If
that completes successfully, adjust the lines to point to the full list of
timesteps:

    -p train-times="$(<  ../train.json)" \
    -p test-times="$(<  ../test.json)" \

Delete any existing GCS outputs (or rename the output directory) and rerun
with `make`.
