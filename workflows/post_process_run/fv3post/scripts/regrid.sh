#!/bin/bash

set -e
set -x


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
gridFiles=2020-11-12-gridspec-orography-and-mosaic-data/$resolution
mosaic=$gridFiles/grid_spec.nc
if [ ! -f $mosaic ]
then
    mkdir -p $gridFiles
    gsutil -m cp gs://vcm-ml-raw/$gridFiles/* $gridFiles
fi
remapFile=${resolution}_to_${nlat}x${nlon}.nc


# fregrid is a GFDL command line tool for regridding FMS model output
# it works for cubed-sphere and many other grids.
fregrid  --input_mosaic $mosaic \
 --remap_file $remapFile \
 --input_file  $localPrefix \
 --output_file $localPrefix.nc \
 --scalar_field $scalarFields \
 $@

gsutil cp $localPrefix.nc $outputBucket
