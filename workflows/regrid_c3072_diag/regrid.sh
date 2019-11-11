#!/bin/bash

set -e
set -x

field="$1"
input_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384
output_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened


# download the data
for tile in {1..6}
do
    fileName=${field}.tile${tile}.nc
    inputUrl=$url/$fileName

    [ -f $fileName ] || gsutil cp $inputUrl $fileName
done

mosaic=2019-10-05-coarse-grids-and-orography-data/C384/grid_spec.nc
nlat=180
nlon=180

fregrid \
 --input_mosaic $mosaic \
 --nlat $nlat --nlon $nlon \
 --remap_file c34_to_${nlat}x${nlon}.nc \
 --input_file  $field \
 --output_file $field.nc  \
 --LstepBegin 1 --LstepEnd 480 \
 --scalar_field $field
 #--scalar_field areat


gsutil cp $field.nc $output_bucket/nLat${nlat}_nLon${nLon}/$field.nc
