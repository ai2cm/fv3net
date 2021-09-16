#!/bin/bash

set -e

wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
tar xzf mpich-3.1.4.tar.gz
cd mpich-3.1.4
./configure --enable-fortran --enable-cxx --prefix=/usr --enable-fast=all,O3
make -j24
make install && ldconfig