#!/bin/sh

data_3d=~/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/3d.zarr
output=data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr

dvc run -o $output -d src/ \
	python -m src.data.regrid $output
