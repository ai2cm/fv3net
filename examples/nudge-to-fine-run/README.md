# Nudging (nudge-to-fine) workflow

This directory contains an example workflow for a nudged-to-fine C48 run to
generate training data for machine learning. 

The configurations are contained in two places:

1. `nudge-to-fine-config.yaml` contains the nudging run configurations
1. `run.sh` contains other configurations, such as the output location, etc.

## Recommended workflow

Edit the configuration files in place. Once your configurations are ready,
submit a test run by running `make nudge_to_fine_run` from the `examples`
directory. If that completes successfully, increase the number of segments to the
desired full length.