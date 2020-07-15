#!/bin/bash

tempdir=$(mktemp -d)
trap 'rm -r $tempdir' EXIT

yq -r '.data["train.json"]' time_configmap.yaml > $tempdir/train.json
yq -r '.data["test.json"]' time_configmap.yaml > $tempdir/test.json

# combine the training and testing times
jq -r -s '[.[][]]' $tempdir/*.json > $tempdir/combined.json

yq . config.yaml > $tempdir/config.json


# insert the times
jq \
    --arg times "$(< $tempdir/combined.json)" \
    --arg train "$(< $tempdir/train.json)"  \
    '.["nudging-times"] |= $times | .["train-times"] |= $train'  \
    $tempdir/config.json > config.json
