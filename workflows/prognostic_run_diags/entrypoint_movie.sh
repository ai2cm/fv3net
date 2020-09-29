#!/bin/bash

set -e

usage="Usage: entrypoint_movie.sh rundir output flags"

if [[ $# != 3 ]]; then
    echo $usage
    exit 2
fi

rundir=$1
output=${2%/}  # strip trailing slash
flags=$3
imageSuffix=_00000.png

hash=$(echo $rundir | md5sum | awk '{print $1}')
workDir=.cache/$hash

python generate_movie_stills.py $flags $rundir $workDir

moviePaths=$(ls $workDir/*$imageSuffix)
for movie in $moviePaths; do
    base=$(basename $movie $imageSuffix)
    ffmpeg -y -r 15 -i $workDir/${base}_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 $workDir/$base.mp4
    gsutil cp $workDir/$base.mp4 $output/$base.mp4
done

