#!/bin/sh

input=data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr
output=data/interim/advection/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr
script=src/data/advect.py 

mkdir -p data/interim/advection

dvc run -o $output -d $script -d $input -f $output.dvc  \
	python $script $input $output
