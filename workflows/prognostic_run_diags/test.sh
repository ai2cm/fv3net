RUNDIR=gs://vcm-ml-scratch/oliwm/2020-06-08-advisory-council-rerun-prognostic/physics-on-rerun/baseline_run_clean
GRID_SPEC=gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec

if [ ! -d "test_rundir" ]; then
    mkdir -p test_rundir
    gsutil -m cp -r $RUNDIR/* test_rundir
fi
if [ ! -d "test_grid_spec" ]; then
    mkdir -p test_grid_spec
    gsutil -m cp -r $GRID_SPEC* test_grid_spec
fi
python save_prognostic_run_diags.py --grid-spec test_grid_spec/grid_spec test_rundir test_diags.nc
