#!/bin/bash

argo submit --from workflowtemplate/restart-prognostic-run \
    -p url="gs://vcm-ml-scratch/test-prognostic-run-example" \
    -p segment-count="2"