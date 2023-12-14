# Reservoir Hyperparemeter Sweeps

This directory contains the hyperparameter sweep code for the reservoir model.

## Running the sweep

To run a sweep, use `submit-sweep.sh`.  This will initialize a sweep over the
specified parameters in `tile-train-sweep.yaml` for each tile and submit a number
of argo jobs (`num_jobs`) specified at the top of the script.

The three training configuration files are:

1. `training-config.yaml` - the main training configuration file
1. `training-data.yaml` - the training data configuration file
1. `validation-data.yaml` - the validation data configuration file

Note that any training configuration parameters specified in the `tile-train-sweep.yaml`
will be override those set in `training-config.yaml` during the sweep.

