NUM_WORKERS=1

python -m fv3net.pipelines.restarts_to_zarr  \
    --init  \
    --runner Direct \
    --num_workers $NUM_WORKERS \
    --n-steps 1  \
    --url gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    --output gs://vcm-ml-data/testing-noah/big_local.zarr