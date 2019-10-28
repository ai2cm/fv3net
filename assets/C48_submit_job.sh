#!/bin/bash

ulimit -s unlimited
cd /FV3/rundir
cp /FV3/fv3.exe /FV3/rundir/fv3.exe
mpirun -np 6 /FV3/rundir/fv3.exe

