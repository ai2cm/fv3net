# submit the job
PIPELINE=fv3net.pipelines.diagnostics_to_zarr
RUNDIR=gs://vcm-ml-data/2019-12-12-baseline-FV3GFS-runs/free/C48/free-2016.93319727-0c4b-4658-8f15-b279b38ae360/output
NUM_WORKERS=5

python -m $PIPELINE  \
    --rundir $RUNDIR \
    --job_name diagnostics-to-zarr \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $NUM_WORKERS \
    --disk_size_gb 100 \
    --worker_machine_type n1-highmem-2
