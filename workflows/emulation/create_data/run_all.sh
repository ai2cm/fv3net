#!/bin/bash
set -e

kustomize build . | kubectl apply -f -

for month in {1..12}
do
   ./run_single.sh $month
done