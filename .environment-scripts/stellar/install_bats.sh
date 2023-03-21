#!/bin/bash
set -e

CLONE_DIR=$1

URL=https://github.com/bats-core/bats-core.git

git clone -b v1.9.0 --depth 1 $URL $CLONE_DIR
