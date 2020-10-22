#!/bin/bash

set -x

run=$1
output=$2
flags=$3

cacheURL=gs://vcm-ml-archive/prognostic_run_diags
cacheKey=`echo $run | cut -c6- | sed 's/\//-/g'`

# check for existence of diagnostics and metrics in cache
gsutil -q stat $cacheURL/$cacheKey/diags.nc
diagsExitCode=`echo $?`
gsutil -q stat $cacheURL/$cacheKey/metrics.json
metricsExitCode=`echo $?`

set -e

if [[ $diagsExitCode -eq 0 && $metricsExitCode -eq 0 ]]; then
    echo "Prognostic run diagnostics detected in cache for given run. Using cached diagnostics."
else
    echo "No prognostic run diagnostics detected in cache for given run. Computing diagnostics and adding to cache."	
    python save_prognostic_run_diags.py $flags $run diags.nc
    python metrics.py diags.nc > metrics.json
    gsutil cp diags.nc $cacheURL/$cacheKey/diags.nc
    gsutil cp metrics.json $cacheURL/$cacheKey/metrics.json	
fi

gsutil cp $cacheURL/$cacheKey/diags.nc $output/diags.nc	
gsutil cp $cacheURL/$cacheKey/metrics.json $output/metrics.json