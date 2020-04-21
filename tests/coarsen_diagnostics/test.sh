set -e
C384_DIAGNOSTICS=gs://vcm-ml-data/test-end-to-end-integration/C384-diagnostics/gfsphysics_15min_coarse.zarr
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/coarsen_diagnostics/

python ../../workflows/coarsen_c384_diagnostics/coarsen_c384_diagnostics.py \
	$C384_DIAGNOSTICS \
	coarsen_c384_diagnostics.yml \
	$OUTPUT
