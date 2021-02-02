cat << EOF > config.yaml
base_version: v0.5
namelist:
  coupler_nml:
    hours: 0
    minutes: 30
  diag_manager_nml:
    flush_nc_files: true
  fv_core_nml:
    do_sat_adj: false
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
  fv_nwp_nudge_nml:
    nudge_hght: false
    nudge_ps: false
    nudge_virt: false
    nudge_winds: false
    nudge_q: false
EOF

echo "{}" >> chunks.yaml


outputUrl=gs://vcm-ml-scratch/noah/tmp/3

python3 prepare_config.py \
  config.yaml \
  gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
  20160801.001500 \
  > fv3config.yml


./run-fv3.sh "$outputUrl" fv3config.yml 2 "$(pwd)/sklearn_runfile.py" chunks.yaml