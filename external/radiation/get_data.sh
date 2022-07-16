#!/bin/bash


#ROOT="${1:-}"

set -e

if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export MYHOME=`pwd`

    if [ ! -d "./fortran" ]; then
        mkdir fortran
        cd fortran
        mkdir data
        mkdir input_data_c12_npz63
        cd data
        mkdir LW
        mkdir SW
        cd $MYHOME
    else
        echo "Fortran output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/data/LW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/lwrad/* ./fortran/data/LW/.
        cd ./fortran/data/LW
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "LW Fortran data already present"
    fi

    if [ -z "$(ls -A ./fortran/data/SW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/swrad/* ./fortran/data/SW/.
        cd ./fortran/data/SW
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "SW Fortran data already present"
    fi
    
    if [ -z "$(ls -A ./fortran/input_data_c12_npz63)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/ML_config/input_data_c12_npz63/* ./fortran/input_data_c12_npz63/.
        cd ./fortran/input_data_c12_npz63
        tar -xzvf dat_files.tar.gz
        cd $MYHOME
    else
        echo "ML Config Fortran data already present"
    fi

    if [ ! -d "./python/lookupdata" ]; then
        cd ./python
        mkdir lookupdata
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz ./lookupdata/.
        cd ./lookupdata
        tar -xzvf lookup.tar.gz
        cd $MYHOME
    else
        echo "Data already present"
    fi

    if [ ! -d "./fortran/radlw/dump" ]; then
        mkdir -p ./fortran/radlw/dump
        cd $MYHOME
    else
        echo "LW standalone output directory already exists"
    fi

    if [ ! -d "./fortran/radsw/dump" ]; then
        mkdir -p ./fortran/radsw/dump
        cd $MYHOME
    else
        echo "SW standalone output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/radlw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/lwrad/* ./fortran/radlw/dump/.
        cd ./fortran/radlw/dump
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "LW standalone data already present"
    fi

    if [ -z "$(ls -A ./fortran/radsw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/swrad/* ./fortran/radsw/dump/.
        cd ./fortran/radsw/dump
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "SW standalone data already present"
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
        mkdir forcing
        cd $MYHOME
    else
        echo "Forcing directory already exists"
    fi

    if [ -z "$(ls -A ./python/forcing)" ]; then
	    gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/* ./python/forcing/.
	    cd ./python/forcing
	    tar -xzvf data.tar.gz
        cd $MYHOME
    else
	    echo "Forcing data already present"
    fi  
    
    echo "Script completed succesfully."
fi
