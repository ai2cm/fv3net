#!/bin/bash

set -e
set -x

nlat=180
nlon=180

field="$1"
input_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384
output_bucket=gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened

# authenticate with gcloud
if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]
then
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
fi

# download the tile data
for tile in {1..6}
do
    fileName=${field}.tile${tile}.nc
    inputUrl=$input_bucket/$fileName

    [ -f $fileName ] || gsutil cp $inputUrl $fileName
done

# download orographic data
mosaic=2019-10-05-coarse-grids-and-orography-data/C384/grid_spec.nc
if [ ! -f $mosaic ]
then
       gsutil cp gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar grid.tar
       tar xf grid.tar
       rm grid.tar
fi


fregrid \
 --input_mosaic $mosaic \
 --nlat $nlat --nlon $nlon \
 --remap_file c34_to_${nlat}x${nlon}.nc \
 --input_file  $field \
 --output_file $field.nc  \
 --scalar_field $field

gsutil cp $field.nc $output_bucket/nLat${nlat}_nLon${nlon}/$field.nc
