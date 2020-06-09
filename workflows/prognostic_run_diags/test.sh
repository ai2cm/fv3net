RUNDIR=gs://vcm-ml-experiments/2020-04-22-advisory-council/deep-off/prognostic_run_clean
GRID_SPEC=gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec

mkdir -p test_rundir
gsutil -m cp -r $RUNDIR test_rundir
python save_prognostic_run_diags.py --grid-spec $GRID_SPEC test_rundir test_diags.nc