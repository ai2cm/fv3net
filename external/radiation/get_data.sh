#!/bin/bash

set -e

mkdir -p data  
if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export MYHOME=`pwd`

    if [ ! -d "./data/lookupdata" ]; then
        cd ./data
        mkdir lookupdata
        cd ./lookupdata
        gsutil cp -r gs://vcm-ml-intermediate/radiation/lookupdata/lookup.tar.gz .
        tar -xzvf lookup.tar.gz
        cd $MYHOME
    else
        echo "Lookup data already present"
    fi
    
    if [ ! -d "./data/forcing" ]; then
        cd ./data
        mkdir -p forcing
        cd ./forcing
	    gsutil -m cp gs://vcm-ml-intermediate/radiation/forcing/* .
	    tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "Forcing data already present"
    fi
fi
