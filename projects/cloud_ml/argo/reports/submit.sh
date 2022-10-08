#!/bin/bash

COMPARISON=$1

name=prognostic-run-report-${COMPARISON}-$(openssl rand -hex 2)
argo submit --from=workflowtemplate/prognostic-run-diags \
    --name $name \
    -p runs="$(< rundirs-${COMPARISON}.json)" \
    -p recompute-diagnostics=false

echo "report generated at: https://storage.googleapis.com/vcm-ml-public/argo/$name/index.html"