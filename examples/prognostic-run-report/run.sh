#!/bin/bash
name=2020-11-09-compare-year-long-prog-runs
argo submit --from=workflowtemplate/prognostic-run-diags \
    --name $name \
    -p docker-image=us.gcr.io/vcm-ml/fv3net:257d6a6319047f6bd78db14e64d42a06e59b310e \
    -p runs="$(< rundirs.json)" \
    -p flags="--verification nudged_c48_fv3gfs_2016"
