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
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz .
        tar -xzvf lookup.tar.gz
        cd $MYHOME
    else
        echo "Lookup data already present"
    fi
    
    if [ ! -d "./data/forcing" ]; then
        cd ./data
        mkdir -p forcing
        cd ./forcing
	    gsutil -m cp gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/* .
	    tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "Forcing data already present"
    fi

    if [ "$USE_DIFFERENT_TEST_CASE" != "" ]; then
      echo "Replacing input data with a different namelist"
      rm -rf ./data/fortran/radiation_driver/*
      gsutil -m cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/ML_config/input_data_c12_npz63_sw_lw/* ./data/fortran/radiation_driver/.
      cd ./data/fortran/radiation_driver
      tar -xvf dat_files.tar.gz
      cd $MYHOME
      cd ./data/lookupdata
      rm rand2d_tile*.nc
      gsutil -m cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/ML_config/random_MLconfig/* .
    fi
fi
