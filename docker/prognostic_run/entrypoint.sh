#!/bin/bash

# install the needed fv3net packages
for package in /fv3net/external/*
do
    if [[ -f $package/setup.py ]]
    then
        echo "Setting up $package"
        pip install -e "$package" --no-deps > /dev/null 2> /dev/null
    elif [[ -f $package/pyproject.toml ]]
    then
        echo "Setting up $package"
        cwd=$PWD
        cd $package
        poetry install --no-dev -n
        cd $cwd
    fi
done

exec "$@"