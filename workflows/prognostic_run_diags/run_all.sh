#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "WARNING: no output_url specified for prognostic run diags."
    exit 1
fi

#gsutil cp rundirs.yml $output_url/rundirs.yml
runs=$(yq . $1)
shift
argo submit argo.yaml -p runs="$runs" $@
