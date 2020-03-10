#!/bin/bash

runs=$(yq . rundirs.yml)
argo submit argo.yaml runs="$runs"
