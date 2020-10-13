#!/bin/bash

set -e 

output=gs://vcm-ml-experiments/noah/2020-09-29-physics-on-free-moreInputs-rev3/prognostic_run
gsutil cp diag_table $output/diag_table

kubectl apply -k ../train-evaluate-prognostic-run/fv3net/workflows/argo
argo submit argo.yaml -p diag_table=$output/diag_table -p output=$output
