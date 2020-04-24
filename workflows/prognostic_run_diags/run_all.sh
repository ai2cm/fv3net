#!/bin/bash

if [ "$#" -lt 1 ]; then
    output_url=gs://vcm-ml-data/experiments-2020-03/prognostic_run_diags
    echo "WARNING: no output_url specified for prognostic run diags."
    echo "Using default output_url $output_url"
else
    output_url=$1
    echo "Saving prognostic run diagnostics to $output_url"
fi

#gsutil cp rundirs.yml $output_url/rundirs.yml
runs=$(yq . rundirs.yml)
argo submit argo.yaml -p runs="$runs" -p output_url="$output_url"
