#!/bin/bash
set -e

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
PLATFORM=$3
COMPILER=$4

URL=https://github.com/NCAR/NCEPlibs.git
SHA=3da51e139d5cd731c9fc27f39d88cb4e1328212b

git clone $URL $CLONE_PREFIX
cd $CLONE_PREFIX
git checkout $SHA
mkdir -p $INSTALL_PREFIX
echo "y" | bash make_ncep_libs.sh -s $PLATFORM -c $COMPILER -d $INSTALL_PREFIX -o 1
