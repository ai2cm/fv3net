# submit the job
PIPELINE=fv3net.pipelines.diagnostics_to_zarr
RUNDIR=$1
NUM_WORKERS=6

python -m $PIPELINE  \
    --rundir $RUNDIR \
    --job_name diags-to-zarr-$(uuidgen) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file workflows/diagnostics_to_zarr/setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $NUM_WORKERS \
    --disk_size_gb 120 \
    --worker_machine_type n1-highmem-4
