# submit the job
PIPELINE=fv3net.pipelines.diagnostics_to_zarr
RUNDIR=$1

python -m $PIPELINE  \
    --rundir $RUNDIR \
    --job_name diags-to-zarr-$(uuidgen) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file workflows/diagnostics_to_zarr/setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 1 \
    --max_num_workers 5 \
    --disk_size_gb 256 \
    --worker_machine_type n1-highmem-16 \
    --extra_package external/vcm/dist/vcm-0.1.1.tar.gz
