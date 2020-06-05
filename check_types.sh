#!/bin/bash

mypy --follow-imports silent external/vcm/vcm/cloud \
external/vcm/vcm/cubedsphere \
fv3net/pipelines/restarts_to_zarr/ \
workflows/fine_res_budget

