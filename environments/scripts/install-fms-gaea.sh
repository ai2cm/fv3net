#!/bin/bash

FMS_DIR=$1

cd $FMS_DIR
autoreconf -f --install
./configure
make -j8 FCFLAGS='-g -FR -i4 -r8'
cp $FMS_DIR/*/*.o $FMS_DIR/*/*.h $FMS_DIR/*/*.mod $FMS_DIR/
