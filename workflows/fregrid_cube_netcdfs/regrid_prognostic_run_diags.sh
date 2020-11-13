#!/bin/bash

input=$1
output=$2

# authenticate with gcloud
if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]
then
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
fi

gsutil cp $input /tmp/prognostic_run_diags.nc
python3 /usr/bin/prognostic_diags_to_tiled.py /tmp/prognostic_run_diags.nc /tmp/diags
fields="$(python3 /usr/bin/print_fields.py /tmp/diags.tile1.nc)"

/usr/bin/regrid.sh /tmp/diags $output C48 $fields --nlon 360 --nlat 180
