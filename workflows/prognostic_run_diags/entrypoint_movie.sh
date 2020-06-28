#!/bin/bash

set -e

rundir=$1
hash=$(echo $rundir | md5sum | awk '{print $1}')

python generate_movie_stills.py $rundir workdir/$hash

ffmpeg -y -r 15 -i workdir/$hash/heating_and_moistening_%05d.png -vf fps=15 -pix_fmt yuv420p -s:v 1920x1080 workdir/$hash/heating_and_moistening.mp4
gsutil cp workdir/$hash/heating_and_moistening.mp4 $output
rm -r workdir/$hash
