#!/bin/bash

# this script orchestrates two workflows to convert diagnostic netCDFs to zarr stores
# and to regrid certain variables to a regular lat-lon grid

# it must be called from root of fv3net repo

DO_REGRID=true
DO_TO_ZARR=true

# constants
ROOT_URL=gs://vcm-ml-data/2020-03-30-learned-nudging-FV3GFS-runs
LATLON_VAR_LIST=DLWRFsfc,DSWRFsfc,DSWRFtoa,LHTFLsfc,PRATEsfc,SHTFLsfc,ULWRFsfc,ULWRFtoa,USWRFsfc,USWRFtoa,TMP2m,TMPsfc,SOILM,PRESsfc,ucomp,vcomp,temp,sphum,ps_dt_nudge,delp_dt_nudge,u_dt_nudge,v_dt_nudge,t_dt_nudge
ARGO_CLUSTER=gke_vcm-ml_us-central1-c_ml-cluster-dev

# what experiments to do post-processing for
RUNS="nudge_mean_T nudge_mean_T_ps nudge_mean_T_ps_u_v"

if [ "$DO_REGRID" = true ]; then
    for RUN in $RUNS; do
        # regrid certain monthly-mean variables to lat-lon grid
        argo --cluster $ARGO_CLUSTER submit workflows/fregrid_cube_netcdfs/pipeline.yaml \
            -p source_prefix=$ROOT_URL/$RUN/atmos_monthly \
            -p output_bucket=$ROOT_URL/$RUN/atmos_monthly.latlon.nc \
            -p fields=$LATLON_VAR_LIST \
            -p extra_args="--nlat 90 --nlon 180"
    done
fi

if [ "$DO_TO_ZARR" = true ]; then
    for RUN in $RUNS; do
        # convert diagnostic output to zarr stores
        python -m fv3net.pipelines.diagnostics_to_zarr  \
            --rundir $ROOT_URL/$RUN \
            --job_name diags-to-zarr-$(uuidgen) \
            --project vcm-ml \
            --region us-central1 \
            --runner DataflowRunner \
            --setup_file workflows/diagnostics_to_zarr/setup.py \
            --temp_location gs://vcm-ml-scratch/tmp_dataflow \
            --num_workers 1 \
            --max_num_workers 5 \
            --disk_size_gb 500 \
            --worker_machine_type n1-highmem-16 \
            --extra_package external/vcm/dist/vcm-0.1.0.tar.gz &
    done
fi
