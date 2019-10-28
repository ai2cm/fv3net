#!/bin/bash

export CFLOW="coarsen-dataflow"
export FV3NET="src"
export DST="scratch/coarsen-dataflow"

mkdir -p $DST

cp -fr $CFLOW/coarseflow $DST/.
cp -fr $CFLOW/setup.py $DST/.
cp -fr $CFLOW/coarseflow_main.py $DST/.
cp -fr $FV3NET $DST/.