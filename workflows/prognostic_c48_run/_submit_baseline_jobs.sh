#!/bin/bash

set -x

STAMP=$(date +%F)-$(uuid | head -c 6)
output=gs://vcm-ml-data/testing-2020-02
urls=("gs://vcm-ml-data/2020-02-28-X-SHiELD-2019-12-02-deep-and-mp-off")
ic=20160803.061500
image=us.gcr.io/vcm-ml/prognostic-run-orchestration:fv3py_v2.3-mp-off-switch

for onestep_url in  "${urls[@]}"
do
    run=$(basename $onestep_url)
    output_url=$output/$run/$STAMP/prognostic_run_baseline
    python orchestrate_submit_job.py -d $onestep_url $output_url $ic $image
done