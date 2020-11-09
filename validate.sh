#!/bin/bash
set -e
set -x

[[ -f ./kustomize ]] || \
    ./install_kustomize.sh 3.8.6

kustomizations=( "examples/" )

for k in $kustomizations; do
    ./kustomize build $k
done
