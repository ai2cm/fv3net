#!/bin/bash

set -e
set -x

dir=2019-09-27-FV3GFS-docker-input-c48-LH-nml
filename=fv3gfs-data-docker_2019-09-27.tar.gz
url=http://storage.googleapis.com/vcm-ml-public/$dir/$filename
datadir_local=inputdata

mkdir -p $datadir_local

# download data
[[ -f $filename ]] || wget $url

# unzip/tar input data
tar xzf $filename -C $datadir_local

rm $filename
mv /inputdata/fv3gfs-data-docker/fix.v201702 /inputdata/fix.v201702  

