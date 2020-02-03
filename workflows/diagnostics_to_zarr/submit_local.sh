#!/bin/bash

PIPELINE=fv3net.pipelines.diagnostics_to_zarr
EXAMPLE_RUNDIR=gs://vcm-ml-data/2019-12-12-baseline-FV3GFS-runs/nudged/C48/test-nudged.935498d5-3528-4e88-b5f4-018e3f54da50/output

# submit the job
python -m $PIPELINE --rundir $EXAMPLE_RUNDIR
