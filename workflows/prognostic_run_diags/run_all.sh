#!/bin/bash

runs=$(yq . rundirs.yml)
argo submit argo.yml runs="$runs"