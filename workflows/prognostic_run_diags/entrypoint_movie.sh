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
imageSuffix=_00000.png

hash=$(echo $rundir | md5sum | awk '{print $1}')
workDir=.cache/$hash

python generate_movie_stills.py $rundir $gridSpec $workDir

moviePaths=$(ls $workDir/*$imageSuffix)
for movie in $moviePaths; do
    base=$(basename $movie $imageSuffix)
    ffmpeg -y -r 15 -i $workDir/${base}_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 $workDir/$base.mp4
    gsutil cp $workDir/$base.mp4 $output/$base.mp4
done

