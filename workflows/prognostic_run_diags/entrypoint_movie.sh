#!/bin/bash

set -e

rundir=$1
output=$2
hash=$(echo $rundir | md5sum | awk '{print $1}')
workdir=.cache/$hash

python generate_movie_stills.py $rundir $workdir
ffmpeg -y -r 15 -i $workdir/heating_and_moistening_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 $workdir/temp.mp4
gsutil cp $workdir/temp.mp4 $output/column_integrated_heating_and_moistening.mp4
rm -r $workdir
