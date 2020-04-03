#!/bin/bash

set -e

# get cluster credentials
cluster_zone=$(gcloud container clusters list --format="value(location)")
cluster_name=$(gcloud container clusters list --format="value(name)")
gcloud container clusters get-credentials --zone $cluster_zone \
    --internal-ip  $cluster_name

exec "$@"