#!/bin/bash

set -e
set -x

nlat=180
nlon=360


src_prefix="$1"; shift;
outputBucket="$1"; shift;
resolution="$1"; shift;
scalarFields="$1"; shift;


# authenticate with gcloud
if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]
then
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
fi

# download the tile data
localPrefix=data
for tile in {1..6}
do
    inputUrl=$src_prefix.tile${tile}.nc
    fileName=$localPrefix.tile${tile}.nc

    [ -f $fileName ] || gsutil cp $inputUrl $fileName
done

# download orographic data
mosaic=2019-10-05-coarse-grids-and-orography-data/$resolution/grid_spec.nc
if [ ! -f $mosaic ]
then
       gsutil cp gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar grid.tar
       tar xf grid.tar
       rm grid.tar
fi
remapFile=${resolution}_to_${nlat}x${nlon}.nc


fregrid  --input_mosaic $mosaic \
 --remap_file $remapFile \
 --nlat $nlat --nlon $nlon \
 --input_file  $localPrefix \
 --output_file $localPrefix.nc \
 --scalar_field $scalarFields

gsutil cp $localPrefix.nc $outputBucket
