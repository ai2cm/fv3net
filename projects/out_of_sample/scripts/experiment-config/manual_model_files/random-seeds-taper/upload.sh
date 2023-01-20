!/bin/bash

upload_dir=gs://vcm-ml-experiments/out-of-sample-right-side-up-winds/oos-configs/oos-full-year-sweep-n2pire-consistent

gsutil cp -r \
    ocsvm-tq-larger-gamma-0-default-tq-0-seed-0 \
    $upload_dir/ocsvm-tq-larger-gamma-0-default-tq-0-seed-0

gsutil cp -r \
    ocsvm-tq-larger-gamma-0-default-tq-0-seed-1 \
    $upload_dir/ocsvm-tq-larger-gamma-0-default-tq-0-seed-1

gsutil cp -r \
    ocsvm-tq-larger-gamma-0-default-tq-0-seed-2 \
    $upload_dir/ocsvm-tq-larger-gamma-0-default-tq-0-seed-2

gsutil cp -r \
    ocsvm-tq-larger-gamma-0-default-tq-0-seed-3 \
    $upload_dir/ocsvm-tq-larger-gamma-0-default-tq-0-seed-3