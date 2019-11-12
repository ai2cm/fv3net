#!/bin/bash

set -e
set -x

function download_mosaic {
    if [ ! -d 2019-10-05-coarse-grids-and-orography-data/ ]
    then
	      gsutil cp -DD gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar grid.tar
	      tar xf grid.tar
	      rm grid.tar
    fi
}

function download_data {
    for tile in {1..6}
    do
        fileName=${field}.tile${tile}.nc
        inputUrl=$url/$fileName

        [ -f $fileName ] || gsutil cp $inputUrl $fileName
    done
}

input_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384
output_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened


download_mosaic
#download_data

mosaic=2019-10-05-coarse-grids-and-orography-data/C384/grid_spec.nc
nlat=180
nlon=360
outputDir=nLat${nlat}_nLon${nlon}

rm -rf $outputDir
mkdir -p $outputDir

for field in $(< fields_to_copy )
do

fregrid \
 --input_mosaic $mosaic \
 --nlat $nlat --nlon $nlon \
 --remap_file c34_to_${nlat}x${nlon}.nc \
 --input_file  $field \
 --output_file $outputDir/$field.nc  \
 --scalar_field $field
 #--scalar_field areat

done

echo gsutil cp -r $outputDir $output_bucket/
