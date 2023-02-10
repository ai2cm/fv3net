#!/usr/bin/env bash

set -e -x


docker run \
	--rm \
    -v /home/mcgibbon/python/fv3net/external:/fv3net/external \
    -v /home/mcgibbon/python/fv3net/workflows:/fv3net/workflows \
    -v /home/mcgibbon/python/fv3net/projects:/fv3net/projects \
    --env-file=../.env \
    -e FSSPEC_GS_REQUESTER_PAYS=vcm-ml \
	-w /fv3net/workflows/prognostic_c48_run \
	us.gcr.io/vcm-ml/post_process_run:b5959bd3d15405e787eafb6fd7db1efa5d00c4bf \
    /bin/bash -c "pip3 install /fv3net/external/vcm && python3 /fv3net/projects/test_fregrid_conservative.py"
