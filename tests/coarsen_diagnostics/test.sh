set -e
C384_DIAGNOSTICS=gs://vcm-ml-data/test-end-to-end-integration/C384-diagnostics/gfsphysics_15min_coarse.zarr
OUTPUT=gs://vcm-ml-data/testing-noah/$(date +%F)/$(git rev-parse HEAD)/coarsen_diagnostics/

local=$(pwd)/tests/coarsen_diagnostics

python workflows/coarsen_c384_diagnostics/coarsen_c384_diagnostics.py \
	$C384_DIAGNOSTICS \
	$local/coarsen_c384_diagnostics.yml \
	$OUTPUT
