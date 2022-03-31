#!/bin/bash

# set 

set -e -x

if [ -z $1 ]; then
    echo "no model passed, exiting"
else
    echo "offline report for model"
    python3 -m fv3net.diagnostics.offline.compute $1 test.yaml $1_offline_diags
    python3 -m fv3net.diagnostics.offline.views.create_report $1_offline_diags $1_offline_report --training-config $1/train.yaml --training-data-config $1/training_data.yaml
fi
# echo "training reference model"
# python3 -m fv3fit.train training-config.yaml train.yaml model_reference --local-download-path train-data-reference
# echo "offline report for reference model"
# python3 -m fv3net.diagnostics.offline.compute model_reference train.yaml offline_diags_reference
# python3 -m fv3net.diagnostics.offline.views.create_report offline_diags_reference offline_report_reference --training-config training-config.yaml --training-data-config train.yaml
