#!/bin/bash

usage="Usage: run.sh <url> [gcs]"

if [[ $# < 1 ]]
then
	echo $usage
fi

cd workflows/prognostic_run_diags


url=$1
(
	export PROG_RUN_LOCATION=$1
	jupyter nbconvert --execute prognostic-run-diags-v1.ipynb
)

if [[ $# > 1 ]]
then
	output=$2
	gsutil cp prognostic-run-diags-v1.html $output/index.html
fi
