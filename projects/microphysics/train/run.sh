#!/bin/bash

set -e

config_file=$1

argo submit argo.yaml \
    -p training-config="$(< $config_file)"