#!/bin/bash
set -e

PREFIX=$1
PLATFORM=$2
INSTALL_PREFIX=$3

cd $PREFIX
./configure $PLATFORM
make -j 8
PREFIX=$INSTALL_PREFIX make install
