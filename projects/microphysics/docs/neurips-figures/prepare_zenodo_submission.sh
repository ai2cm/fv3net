#!/bin/bash

set -ex

function uploadToZenodo {
    # from https://github.com/jhpoelen/zenodo-upload; MIT license
    # strip deposition url prefix if provided; see https://github.com/jhpoelen/zenodo-upload/issues/2#issuecomment-797657717
    DEPOSITION=$( echo $1 | sed 's+^http[s]*://zenodo.org/deposit/++g' )
    FILEPATH="$2"
    FILENAME=$(echo $FILEPATH | sed 's+.*/++g')
    BUCKET=$(curl https://zenodo.org/api/deposit/depositions/"$DEPOSITION"?access_token="$ZENODO_TOKEN" | jq --raw-output .links.bucket)
    curl --progress-bar -o /dev/null --upload-file "$FILEPATH" $BUCKET/"$FILENAME"?access_token="$ZENODO_TOKEN"
}

function downloadRandomSubset {
    src=$1
    dest=$2
    n=$3

    mkdir -p $dest
    gsutil ls $src | shuf | head -n $n | gsutil -m cp -I $dest
}
DATA=gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4

if ! [[ -d zenodo/models ]]
then
    mkdir -p zenodo/models
    files="gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1 \
        gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1 \
        gs://vcm-ml-experiments/microphysics-emulation/2022-07-19/precpd-diff-only-rnn-v1-shared-weights-v1"
    for file in $files
    do
        gsutil -m cp -r $file zenodo-folder/models
    done
fi

if ! [[ -d zenodo/data/train ]]
then
    downloadRandomSubset $DATA/train zenodo/data/train 1000
fi

if ! [[ -d zenodo/data/test ]]
then
    downloadRandomSubset $DATA/test zenodo/data/test 200
fi

rm -rf zenodo/plot-data
mkdir -p zenodo/plot-data
cp *.nc *.csv zenodo/plot-data/

if ! [[ -d data.zip ]]
then
    zip -0 -r data.zip zenodo
fi

uploadToZenodo 7109065 data.zip