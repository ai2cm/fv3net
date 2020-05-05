#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "WARNING: no rundirs.json specified for prognostic run diags."
    exit 1
fi

runs=$(cat $1)
shift
argo submit argo.yaml -p runs="$runs" $@
