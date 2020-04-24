#!/bin/bash -e


function downloadTiles {
	baseName=$(basename $2)
	if [[ ! -f "$baseName.tile1.nc" ]]; then
	gsutil -m cp "$1.tile?.nc" .
	else
		echo "$baseName already exists locally"
	fi
}

usage="Usage: entrypoint.sh [ -g grid_spec_path ] rundir output"


while getopts "g:" OPTION; do
	case $OPTION in 
		g)
			shift
			gridSpec=$OPTARG
			;;
		*)
		echo $usage
		exit 1
		;;
	esac
done

if [[ $# != 2 ]]; then
	echo $usage
fi


rundir=$1
output=$2

# make a local working directory based on input hash
# this will allow this script to be resumable
localWorkDir=$(md5 -q -s $rundir)
mkdir -p $localWorkDir
(
	cd $localWorkDir
	gridSpec=gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec

	downloadTiles $1/atmos_dt_atmos atmos_dt_atmos
	downloadTiles $gridSpec grid_spec

	[[ -f diags.nc ]] || python ../save_prognostic_run_diags.py --grid-spec $gridSpec ./ diags.nc
	python ../metrics.py diags.nc > metrics.json

	gsutil cp diags.nc $output/diags.nc
	gsutil cp metrics.json $output/metrics.json
)