#!/bin/bash

# fv3gfs-fortran arguments
FV3GFS_PLATFORM=gnu_docker

# Python requirements environment variables
MPI4PY_CC=/usr/bin/mpicc
MPI4PY_MPICC=/usr/bin/mpicc

# Python wrapper environment variables
WRAPPER_CC='x86_64-linux-gnu-gcc -pthread'
WRAPPER_LDSHARED='x86_64-linux-gnu-gcc -pthread -shared'
