#!/bin/bash

set -e

# build fv3
#make -C /fv3net/external/fv3gfs-fortran/FV3
#PREFIX=/usr/local/ make -C /fv3net/external/fv3gfs-fortran/FV3 install

# build fv3gfs-wrapper
# rm -f /fv3net/external/fv3gfs-wrapper/lib/*.o
# make -C /fv3net/external/fv3gfs-wrapper build

# run the model


runfile=$(pwd)/sklearn_runfile.py
cd rundir/
mpirun -n 24 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python -m mpi4py "$runfile"
#mpirun -n 24 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none fv3.exe

