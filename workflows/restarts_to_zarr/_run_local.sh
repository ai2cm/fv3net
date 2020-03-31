NUM_WORKERS=1

cd /home/noahb/fv3net

wheel_args="--setup $(pwd)/setup.py"
for wheel in $(find dist -name '*.whl')
do 
    wheel_args="$wheel_args --extra_package $wheel"
done

echo $wheel_args

python -m fv3net.pipelines.restarts_to_zarr  \
    --init  \
    --runner Direct \
    --num_workers $NUM_WORKERS \
    --n-steps 1  \
    --url gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    --output gs://vcm-ml-data/testing-noah/big_local.zarr