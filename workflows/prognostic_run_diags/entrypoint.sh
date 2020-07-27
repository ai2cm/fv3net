#!/bin/bash -e

set -e

function downloadTiles() {
    baseName=$(basename $2)
    if [[ ! -f "$baseName.tile1.nc" ]]; then
        gsutil -m cp "$1.tile?.nc" .
    else
        echo "$baseName already exists locally"
    fi
}

function downloadZarr() {
    baseName=$(basename $1)
    if [[ ! -d "$baseName" ]]; then
        gsutil -m cp -r "$1" . || echo "No diagnostics zarr found at $1"
    else
        echo "$baseName already exists locally"
    fi
}

usage="Usage: entrypoint.sh rundir output"

if [[ $# != 2 ]]; then
    echo $usage
    exit 2
fi

[[ -n $GOOGLE_APPLICATION_CREDENTIALS ]] && gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS

rundir=$1

# strip trailing slash
output=${2%/}

cwd=$PWD

# make a local working directory based on input hash
# this will allow this script to be resumable
localWorkDir=.cache/$(echo $rundir | md5sum | awk '{print $1}')
mkdir -p $localWorkDir

cd $localWorkDir

downloadZarr $1/atmos_dt_atmos.zarr
downloadZarr $1/sfc_dt_atmos.zarr
downloadZarr $1/diags.zarr

[[ -f diags.nc ]] || python $cwd/save_prognostic_run_diags.py ./ diags.nc
python $cwd/metrics.py diags.nc >metrics.json

gsutil cp diags.nc $output/diags.nc
gsutil cp metrics.json $output/metrics.json
