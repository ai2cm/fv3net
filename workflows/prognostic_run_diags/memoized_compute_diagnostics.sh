#!/bin/bash

set -x

run=$1
output=$2
flags=$3

cacheKey=$(echo "$run" | cut -c6- | sed 's/\//-/g')
cacheURL=gs://vcm-ml-archive/prognostic_run_diags/$cacheKey

# check for existence of diagnostics and metrics in cache
gsutil -q stat "$cacheURL/diags.nc"
diagsExitCode=$?
gsutil -q stat "$cacheURL/metrics.json"
metricsExitCode=$?

set -e

if [[ $diagsExitCode -eq 0 && $metricsExitCode -eq 0 ]]; then
    echo "Prognostic run diagnostics detected in cache for given run. Using cached diagnostics."
else
    echo "No prognostic run diagnostics detected in cache for given run. Computing diagnostics and adding to cache."	
    python save_prognostic_run_diags.py $flags $run diags.nc
    python metrics.py diags.nc > metrics.json
    gsutil cp diags.nc "$cacheURL/diags.nc"
    gsutil cp metrics.json "$cacheURL/metrics.json"
fi

gsutil cp "$cacheURL/diags.nc" "$output/diags.nc"
gsutil cp "$cacheURL/metrics.json" "$output/metrics.json"