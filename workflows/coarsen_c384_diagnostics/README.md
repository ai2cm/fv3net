#### Usage
Run `python coarse_grain.py` from `fv3net/workflows/coarse_grain_c384_diags`.

#### Description
This script coarsens a subset of diagnostic variables from the high resolution runs
and saves them as a zarr in C48 resolution. The output of this workflow is used later
as an input to the training data pipeline, which merges in this data to be used as
features for training the ML model. 

In the future we can move this task into the SHiELD post-processing and 
upload workflow.