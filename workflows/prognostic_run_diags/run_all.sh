#!/bin/bash

gsutil cp rundirs.yml gs://vcm-ml-data/testing-2020-02/rundirs.yml
runs=$(yq . rundirs.yml)
argo submit argo.yaml -p runs="$runs"
