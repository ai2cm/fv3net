#!/bin/bash

# install the needed fv3net packages
for package in /external/*
do
    echo "Setting up $package"
    pip install -e "$package" --no-deps > /dev/null
done

echo "Setting up /fv3fit"
pip install --no-deps -e /fv3fit > /dev/null

exec "$@"