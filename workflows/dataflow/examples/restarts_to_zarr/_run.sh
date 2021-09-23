    NUM_WORKERS=256


./dataflow.sh submit  -m fv3net.pipelines.restarts_to_zarr  \
        gs://vcm-ml-intermediate/2021-08-06-PIRE-c48-restarts-post-spinup \
        gs://vcm-ml-intermediate/2021-09-22-PIRE-c48-restarts-post-spinup-to-zarr/pire_c48_post_spinup_restarts.zarr \
        --select-variables T u v W DZ delp phis \
        --select-daily-times 000000 030000 060000 090000 120000 150000 180000 210000 \
        --select-categories fv_core_coarse.res \
        --no-coarse-suffix \
        --runner DataflowRunner \
        --project vcm-ml \
        --region us-central1 \
        --setup_file /home/AnnaK/fv3net/workflows/dataflow/setup.py \
        --job_name save-pire-restarts-3h \
        --temp_location gs://vcm-ml-scratch/tmp_dataflow \
        --num_workers 128 \
        --autoscaling_algorithm=NONE \
        --worker_machine_type n1-standard-1 \
        --disk_size_gb 30 \
        --extra_package /home/AnnaK/fv3net/external/vcm/dist/vcm-0.1.0.tar.gz \
        --extra_package /home/AnnaK/fv3net/external/vcm/external/mappm/dist/mappm-0.1.0.tar.gz
            # --n-steps 50  \