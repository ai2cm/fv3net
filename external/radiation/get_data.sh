#!/bin/bash

set -e

mkdir -p fortran  
if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export MYHOME=`pwd`

    if [ ! -d "./fortran/data" ]; then
        mkdir -p fortran
        cd ./fortran
        mkdir -p data
        cd data
        cd $MYHOME
    else
        echo "Fortran output directory already exists"
    fi

    cd ./python
    mkdir -p lookupdata
    cd $MYHOME

    if [ -z "$(ls -A ./python/lookupdata)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz ./python/lookupdata/.
        cd ./python/lookupdata
        tar -xzvf lookup.tar.gz
        cd $MYHOME
    else
        echo "Data already present"
    fi

    if [ -z "$(ls -A ./fortran/data/radiation_driver)" ]; then
        gsutil -m cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/radiation_driver/ ./fortran/data/.
        cd ./fortran/data/radiation_driver
        tar -xzvf dat_files.tar.gz
        cd $MYHOME
    else
        echo "Driver standalone data already present"
    fi

    if [ ! -d "./python/forcing" ]; then
        cd ./python
        mkdir -p forcing
        cd $MYHOME
    else
        echo "Forcing directory already exists"
    fi

    if [ -z "$(ls -A ./python/forcing)" ]; then
	    gsutil -m cp gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/* ./python/forcing/.
	    cd ./python/forcing
	    tar -xzvf data.tar.gz
        cd $MYHOME
    else
	    echo "Forcing data already present"
    fi  

    if [ "$USE_DIFFERENT_TEST_CASE" != "" ]; then
      echo "Replacing input data with a different namelist"
      rm -rf ./fortran/data/radiation_driver/*
      gsutil -m cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/ML_config/input_data_c12_npz63_sw_lw/* ./fortran/data/radiation_driver/.
      cd ./fortran/data/radiation_driver
      tar -xvf dat_files.tar.gz
      cd $MYHOME
      cd ./python/lookupdata
      rm rand2d_tile*.nc
      gsutil -m cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/ML_config/random_MLconfig/* .
    fi
fi
