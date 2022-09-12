#!/bin/bash
set -e

ROOT_DIRECTORY=$1

cd $ROOT_DIRECTORY
pip install --no-cache-dir -r docker/prognostic_run/requirements.txt
pip install --no-cache-dir --no-dependencies -e external/vcm
pip install --no-cache-dir --no-dependencies -e external/loaders
pip install --no-cache-dir --no-dependencies -e external/fv3fit
pip install --no-cache-dir --no-dependencies -e external/fv3kube
pip install --no-cache-dir --no-dependencies -e workflows/post_process_run
pip install --no-cache-dir --no-dependencies -e workflows/prognostic_c48_run
pip install --no-cache-dir --no-dependencies -e external/emulation
pip install --no-cache-dir --no-dependencies -e external/radiation
