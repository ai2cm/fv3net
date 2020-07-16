#!/bin/bash

## Calls entrypoint.sh followed by submit_workflow.sh after consolidating config

# https://stackoverflow.com/questions/4632028/how-to-create-a-temporary-directory
export CONFIG=`mktemp -d -t`

function cleanup {
    rm -rf ${CONFIG}
    echo "Deleted temp working directory ${CONFIG}, time config and end-to-end.yml"
    rm tmpconfig.json train_and_test_times.json one_step_times.json end-to-end.yml
}

trap cleanup EXIT

# Copy base configuration files
cp ./workflows/end_to_end/kustomization/* ${CONFIG}/.

# Copy e2e configuration files with overwrite
cp -f ./tests/end_to_end_integration/kustomization/* ${CONFIG}/.

export $(xargs <${CONFIG}/input_data.env) 

./workflows/end_to_end/entrypoint.sh



