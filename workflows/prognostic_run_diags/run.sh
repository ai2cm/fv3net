#!/bin/bash

usage="Usage: run.sh <url> [gcs]"

case $# in 
  1) 
  input=$1
  output=gs://vcm-ml-public/${input##gs://vcm-ml-data/}
  ;;
  2)
  input=$1 
  output=$2
  ;;
  *)
  echo $usage
  ;;
esac

cd workflows/prognostic_run_diags

python savediags.py $input diags.nc || exit -1
jupyter nbconvert --execute index.ipynb

gsutil cp index.html $output/index.html
gsutil acl ch -u AllUsers:R $output/index.html
echo http://storage.googleapis.com/${output##gs://}/index.html
