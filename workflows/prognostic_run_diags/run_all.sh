#!/bin/bash

runs=$(yq . rundirs.yml)
argo submit argo.yaml -p runs="$runs"
gsutil cp rundirs.yml gs://vcm-ml-data/testing-2020-02/rundirs.yml
