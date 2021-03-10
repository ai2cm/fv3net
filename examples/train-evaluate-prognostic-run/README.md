# Training and Prognostic run workflow

This directory contains an example workflow for running the end to end
workflow starting with pre-computed training data. It performs the following
operations:
- train one or more models
- offline report
- prognostic run using the model(s) trained in step 1
- compute prognostic run diagnostic netCDFs and json files. A report combining
  several runs should be run separately.

The configurations are contained in three places:

1. `prognostic-run.yaml` contains the prognostic run configurations
2. `training-config.yaml` contains the ML model(s) settings. Note the required format:
a list of dicts, each with entries `name` and `config`. If only one model is to be trained
and used, just provide a list of length one.
3. `run.sh` contains other configurations, such as the output location,
   timestep selection, etc.


## Recommended workflow

Edit the configuration files in place. To ease development, `run.sh`
currently points to shortend lists of training/testing times in these lines:

    -p train-times="$(<  ../../train_short.json)" \
    -p test-times="$(<  ../../test_short.json)" \


Once your configurations are ready, submit a testing run by running 
`make submit_train_evaluate_prognostic_run` from the `examples` directory. If
that completes successfully, adjust the lines to point to the full list of
timesteps:

    -p train-times="$(<  ../../train.json)" \
    -p test-times="$(<  ../../test.json)" \

Delete any existing GCS outputs (or rename the output directory) and rerun.
