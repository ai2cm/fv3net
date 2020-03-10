#!/bin/bash

runs=$(yq . rundirs.yml)
argo submit argo.yaml -p runs="$runs"
