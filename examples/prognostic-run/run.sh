#!/bin/bash

set -e 

output=gs://vcm-ml-scratch/noah/test-free
gsutil cp diag_table $output/diag_table

kubectl apply -k ../train-evaluate-prognostic-run/fv3net/workflows/argo
argo submit argo.yaml -p diag_table=$output/diag_table -p output=$output
