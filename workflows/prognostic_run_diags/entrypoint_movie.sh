#!/bin/bash

set -e

usage="Usage: entrypoint_movie.sh rundir output"

if [[ $# != 2 ]]; then
    echo $usage
    exit 2
fi

rundir=$1
output=$2
gridSpec=gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec

hash=$(echo $rundir | md5sum | awk '{print $1}')
workdir=.cache/$hash

python generate_movie_stills.py $rundir $gridSpec $workdir
ffmpeg -y -r 15 -i $workdir/heating_and_moistening_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 $workdir/movie.mp4
gsutil cp $workdir/movie.mp4 $output/column_integrated_heating_and_moistening.mp4

