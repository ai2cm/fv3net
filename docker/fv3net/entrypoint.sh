#!/bin/bash

set -e

if [[ ! -z $GOOGLE_APPLICATION_CREDENTIALS ]]; then
    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
fi

# get cluster credentials
cluster_zone=$(gcloud container clusters list --format="value(location)")
cluster_name=$(gcloud container clusters list --format="value(name)")
gcloud container clusters get-credentials --zone $cluster_zone \
    --internal-ip  $cluster_name

exec "$@"