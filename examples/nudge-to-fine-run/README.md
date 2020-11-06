# Nudging (nudge-to-fine) workflow

This directory contains an example workflow for a nudged-to-fine C48 run to
generate training data for machine learning. 

The configurations are contained in three places:

1. `nudging-run.yaml` contains the nudging run configurations
2. `kustomize.yaml` specifies image names and tags
1. `run.sh` contains other configurations, such as the output location,
   timestep selection, etc.

## Recommended workflow

Edit the configuration files in place. Once your configurations are ready,
submit a test run by running `make`. If that completes successfully, adjust the
lines to point to the full list of timesteps:

    -p times="$(< output-times.json)"

Also increase the number of segments to the desired full length. Delete any existing
GCS outputs (or rename the output directory) and rerun with `make`.

## Updating the fv3net submodule

To take advantage of upstream changes to the argo workflows you can update
the `fv3net` submodules in this folder.