#!/bin/bash

usage="Usage: run.sh <url> [gcs]"

if [[ $# < 1 ]]
then
	echo $usage
	exit -1
fi

url=$1
cd workflows/prognostic_run_diags

python savediags.py $url diags.nc || exit -1
jupyter nbconvert --execute index.ipynb

if [[ $# > 1 ]]
then
	output=$2
	gsutil cp index.html $output/index.html
	gsutil acl ch -u AllUsers:R $output/index.html
	echo http://storage.googleapis.com/${output##gs://}/index.html
fi
