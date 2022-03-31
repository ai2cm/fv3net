#!/bin/bash

# set -e

# BUCKET=vcm-ml-scratch # don't pass bucket arg to use default 'vcm-ml-experiments'
# PROJECT=oliwm # don't pass project arg to use default 'default'
# # TAG is the primary way by which users query for experiments with artifacts
# # it determines the output directory for the data
# TAG=fine-res-train-test-offline
# NAME=${TAG}-$(openssl rand --hex 6) # required

# argo submit --from workflowtemplate/train-diags-prog \
#     -p bucket=${BUCKET} \
#     -p project=${PROJECT} \
#     -p tag=${TAG} \
#     -p training-configs="$( yq . training-config.yaml )" \
#     -p training-data-config="$( yq . train.yaml )" \
#     -p test-data-config="$( yq . test.yaml )" \
#     -p validation-data-config="$( yq . validation.yaml )" \
#     -p training-flags="--cache.local_download_path train-data-download-dir" \
#     -p prognostic-run-config="$(< prognostic-run.yaml)" \
#     -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$TAG \
#     -p cpu-training=7500m \
#     -p memory-training=24Gi \
#     -p segment-count=1 \
#     -p do-prognostic-run="false" \
#     --name "${NAME}"

set -e -x

if [ ! -d cached_train ]; then
    echo "caching data"
    ./cache_data.sh
fi
echo "training model"
python -m fv3fit.train training-config.yaml cached-train.yaml model --validation-data-config cached-validation.yaml
# fil-profile run -m fv3fit.train train-dense.yaml cached_data.yaml model_1batch
# echo "offline report for 1batch model"
# python3 -m fv3net.diagnostics.offline.compute model_1batch train.yaml offline_diags_1batch
# python3 -m fv3net.diagnostics.offline.views.create_report offline_diags_1batch offline_report_1batch --training-config training-config-1batch.yaml --training-data-config train_small_1batch.yaml

# echo "training reference model"
# python3 -m fv3fit.train training-config.yaml train.yaml model_reference --local-download-path train-data-reference
# echo "offline report for reference model"
# python3 -m fv3net.diagnostics.offline.compute model_reference train.yaml offline_diags_reference
# python3 -m fv3net.diagnostics.offline.views.create_report offline_diags_reference offline_report_reference --training-config training-config.yaml --training-data-config train.yaml
