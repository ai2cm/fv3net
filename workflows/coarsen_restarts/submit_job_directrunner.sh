#!/bin/sh

GCS_SRC="gs://vcm-ml-data/2019-12-02-40-day-X-SHiELD-simulation-C384-restart-files"
GCS_DST="gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts"
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy"

SRC_RESOLUTION=384
TARGET_RESOLUTION=48

python -m fv3net.pipelines.coarsen_restarts\
    --gcs-src-dir $GCS_SRC \
    --gcs-dst-dir $GCS_DST \
    --gcs-grid-spec-path $GCS_GRIDSPEC \
    --source-resolution $SRC_RESOLUTION \
    --target-resolution $TARGET_RESOLUTION \
    --runner DirectRunner 