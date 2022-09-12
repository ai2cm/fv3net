#!/bin/bash
set -e

PREFIX=$1

URL=https://github.com/nbren12/call_py_fort.git
BRANCH=v0.2.0
git clone $URL --branch=$BRANCH $PREFIX
cd $PREFIX
make
make install
ldconfig
