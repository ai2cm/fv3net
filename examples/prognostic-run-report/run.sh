#!/bin/bash
name=2020-12-17-n2f-moisture-conservation-$(openssl rand -hex 6)
argo submit --from=workflowtemplate/prognostic-run-diags \
    --name $name \
    -p runs="$(< rundirs.json)" \
    -p make-movies=false

echo "report generated at: https://storage.googleapis.com/vcm-ml-public/argo/$name/index.html"
