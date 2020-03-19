#!/bin/bash

set -x
set -e

ROOT=$PWD

if [[ -z "$FV3GFS_PYTHON_DIR" ]]; then
    echo "Must provide FV3GFS_PYTHON_DIR in environment" 1>&2
    exit 1
fi

cd $FV3GFS_PYTHON_DIR
./build_docker.sh && docker tag us.gcr.io/vcm-ml/fv3gfs-python:latest us.gcr.io/vcm-ml/fv3gfs-python:v0.3.1
cd $ROOT
make || tail -n 100 outdir/stderr.log | less
