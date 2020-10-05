#!/bin/bash

set -e

usage="Usage: stitch_movie_stills.sh workDir output"

if [[ $# != 2 ]]; then
    echo $usage
    exit 2
fi

workDir=$1
output=${2%/}  # strip trailing slash

imageSuffix=_00000.png

moviePaths=$(ls $workDir/*$imageSuffix)
for movie in $moviePaths; do
    base=$(basename $movie $imageSuffix)
    ffmpeg -y -r 15 -i $workDir/${base}_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 $workDir/$base.mp4
    gsutil cp $workDir/$base.mp4 $output/$base.mp4
done

