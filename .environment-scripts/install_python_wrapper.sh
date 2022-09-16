#!/bin/bash
set -e

FV3_DIR=$1

make -C $FV3_DIR wrapper_build
pip install --no-dependencies $FV3_DIR/wrapper/dist/*.whl
