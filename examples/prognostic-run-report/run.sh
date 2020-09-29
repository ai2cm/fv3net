#!/bin/bash
name=$(date +%Ft%H%M%S)-$(openssl rand -hex 6)
argo submit --from=wftmpl/prognostic-run-diags \
    --name $name \
    -p docker-image=us.gcr.io/vcm-ml/fv3net:c17c8080ec873d1f8e0e93463eaa24148cb172ee \
    -p runs="$(< rundirs.json)" \
    -p make-movies=true
