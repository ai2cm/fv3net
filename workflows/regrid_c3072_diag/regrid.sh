#!/bin/bash

field=PRATEsfc

for tile in {1..6}
do
    outputFile=$field.tile${tile}.nc
    rm $outputFile
    mppnccombine $outputFile $field/*tile${tile}.nc.* 
done

mosaic=2019-10-05-coarse-grids-and-orography-data/C384/grid_spec.nc

fregrid \
 --input_mosaic $mosaic \
 --nlat 180 --nlon 90 \
 --remap_file c34_to_180x90.nc \
 --input_file  $field \
 --output_file $field.nc  \
 --LstepBegin 1 --LstepEnd 480 \
 --scalar_field $field
 #--scalar_field areat
