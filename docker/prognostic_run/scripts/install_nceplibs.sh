#!/bin/bash
set -e

PREFIX="$1"

curl -L https://github.com/NCAR/NCEPlibs/archive/3da51e139d5cd731c9fc27f39d88cb4e1328212b.tar.gz | tar xz
mkdir -p "$PREFIX"
cd NCEPlibs-*
echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d "$PREFIX" -o 1