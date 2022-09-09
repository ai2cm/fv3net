#!/bin/bash
set -e

INSTALL_PREFIX=$1

cd $INSTALL_PREFIX
autoreconf -f --install
./configure
make -j8 $FLAGS
mv $INSTALL_PREFIX/*/*.o $INSTALL_PREFIX/*/*.h $INSTALL_PREFIX/*/*.mod $INSTALL_PREFIX/
