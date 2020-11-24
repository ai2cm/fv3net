#!/bin/bash

set -xe

input=$1
output=$2

# authenticate with gcloud
if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]
then
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
fi

gsutil cp $input /tmp/data.nc
single_netcdf_to_tiled.py /tmp/data.nc /tmp/tiled_data
fields="$(print_fields.py /tmp/tiled_data.tile1.nc)"

fregrid_cubed_to_latlon.sh /tmp/tiled_data $output C48 $fields --nlon 360 --nlat 180

rm /tmp/data.nc /tmp/tiled_data.tile?.nc