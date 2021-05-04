# Nudging (nudge-to-fine) workflow

This directory contains an example workflow for a nudged-to-fine C48 run to
generate training data for machine learning. 

The configurations are contained in two places:

1. `nudging-run.yaml` contains the nudging run configurations
1. `run.sh` contains other configurations, such as the output location,
   timestep selection, etc.
   
Output will be written to "gs://{bucket}/{project}/$(date +%F)/{tag}/fv3gfs_run".

## Recommended workflow

Edit the configuration files in place. You may submit a test run by specifying 
`-p segment-count="1"` and `-p bucket=vcm-ml-scratch`, for example. When re-running
a job, delete any existing GCS outputs (or rename the output directory).
