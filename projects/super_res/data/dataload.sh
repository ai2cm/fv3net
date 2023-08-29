#! /bin/sh
channel='c48_atmos_ave'
file='atmos_8xdaily_ave_coarse.zarr'
for member in $(seq -f "%04g" 1 11)
do
    mkdir -p /data/prakhars/ensemble/$channel/$member
    gsutil -m cp -r gs://vcm-ml-raw-flexible-retention/2023-08-14-C384-reference-ensemble/ic_$member/diagnostics/$file /data/prakhars/ensemble/$channel/$member
done
# channel --> file
# c384_precip_ave --> sfc_8xdaily_ave.zarr
# c48_precip_plus_more_ave --> sfc_8xdaily_ave_coarse.zarr
# c384_topo --> atmos_static.zarr
# c48_atmos_ave --> atmos_8xdaily_ave_coarse.zarr