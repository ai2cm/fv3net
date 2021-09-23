NUM_WORKERS=1

python -m fv3net.pipelines.restarts_to_zarr  \
    gs://vcm-ml-scratch/annak/2021-08-04-correct-dtype-pire-coarsened-restarts \
    gs://vcm-ml-scratch/annak/2021-09-21-test-restarts-to-zarr/test.zarr \
    --no-coarse-suffix \
    --select-variables T u v W DZ delp phis \
    --select-daily-times 090000 150000 \
    --runner Direct \
    --num_workers 1
   # --n-steps 1  
