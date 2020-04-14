#!/bin/bash

set -e

if [[ ! -z $GOOGLE_APPLICATION_CREDENTIALS ]]; then
    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
fi

gsutilPath=$(which gsutil)
echo "Path to gsutil: $gsutilPath"
if [[ "$gsutilPath" != /usr/bin/gsutil ]]; then
    echo "gsutil is not the system version...which could lead to auth issues. deleting the package."
    rm $gsutilPath
fi

exec "$@"
