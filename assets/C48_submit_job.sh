#!/bin/bash

ulimit -s unlimited
cd /FV3/rundir
cp /FV3/fv3.exe /FV3/rundir/fv3.exe
mpirun -np 6 --allow-run-as-root  --mca btl_vader_single_copy_mechanism none --oversubscribe /FV3/rundir/fv3.exe

