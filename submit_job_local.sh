#!/bin/bash

NUMCODECS_WHEEL=/home/noahb/tmp/numcodecs-0.6.4-cp37-cp37m-linux_x86_64.whl
root=$(pwd)

set -x

poetry_packages=(
  . 
  external/vcm 
  external/vcm/external/mappm
  workflows/fine_res_budget
)

set -e

function buildSdist {
  (
    cd "$1"
    rm -rf dist
    python setup.py sdist
    cp dist/*.tar.gz "$2"
  )
}

function buildPackages {
  rm -rf $1
  mkdir -p $1
  for package in "${poetry_packages[@]}"
  do
    buildSdist "$package" $1
  done
}


workdir=$(mktemp -d)
buildPackages $workdir/dists/
cp $NUMCODECS_WHEEL $workdir/dists/

cd $workdir
echo "Running in $workdir"

function runLocalIsolate {
  python -m venv env
  source env/bin/activate
  pip install dists/mappm*.tar.gz
  pip install dists/numcodecs*.whl
  pip install dists/vcm*.tar.gz
  pip install dists/budget*.tar.gz
}



function runRemote {
  extraPackages=( dists/*.tar.gz )

  packageArgs=""
  for pkg in "${extraPackages[@]}"
  do
    packageArgs+=" --extra_package $pkg"
  done
  
  
  cmd="python $@ $packageArgs"
  echo "Running: $cmd"
  $cmd
}

runRemote $@
