#!/bin/bash

#############
# Run the end-to-end integration workflow from a local machine.
# Call script from the fv3net directory.  The user argument for an end-to-end
# configuration file is optional, but if specified will be used in place of the default
# integration test configuration.
#############

end_to_end_config=$1

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

# If user provided an end-to-end prioritize that by copying last
echo $end_to_end_config
if [ -f $end_to_end_config ]; then
    echo Copying user-supplied end-to-end configuration ${end_to_end_config}
    cp -f $end_to_end_config ${CONFIG}/end_to_end.yaml
fi

export $(xargs <${CONFIG}/input_data.env) 

./workflows/end_to_end/entrypoint.sh
