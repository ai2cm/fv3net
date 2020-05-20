#!/bin/bash


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
    rm -rd dist
    python setup.py sdist >> /dev/null
    cp dist/*.tar.gz "$2"
  )
}

function buildPackages {
  rm -rf dists/
  mkdir -p dists
  for package in "${poetry_packages[@]}"
  do
    buildSdist "$package" "$(pwd)/$1"
  done
}

buildPackages dists/
extraPackages=( dists/*.tar.gz )

packageArgs=""
for pkg in "${extraPackages[@]}"
do
  packageArgs+=" --extra_package $pkg"
done


cmd="python $@ $packageArgs"
echo "Running: $cmd"
$cmd