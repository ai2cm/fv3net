#!/bin/bash
name=$(date +%Ft%H%M%S)-$(openssl rand -hex 6)
argo submit --from=workflowtemplate/prognostic-run-diags \
    --name $name \
    -p docker-image=us.gcr.io/vcm-ml/fv3net:9ca9d9bd4bf37b2b4ffcdc195dcae63580f8694e \
    -p runs="$(< rundirs.json)" \
    -p make-movies=true

echo "report generated at: https://storage.googleapis.com/vcm-ml-public/argo/$name/index.html"
