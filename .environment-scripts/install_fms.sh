#!/bin/bash
set -e

INSTALL_PREFIX=$1
MAKE_OPTIONS=$2

cd $INSTALL_PREFIX
autoreconf -f --install
./configure
make $MAKE_OPTIONS
mv $INSTALL_PREFIX/*/*.o $INSTALL_PREFIX/*/*.h $INSTALL_PREFIX/*/*.mod $INSTALL_PREFIX/
