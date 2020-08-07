#!/bin/bash

config_input="$1"
times_input="$2"

tempdir=$(mktemp -d)
trap 'rm -r $tempdir' EXIT

yq . "$config_input" > $tempdir/config.json

# insert the times
jq \
    --arg times "$(< $times_input)" \
    '.["train-times"] |= $times'  \
    $tempdir/config.json
