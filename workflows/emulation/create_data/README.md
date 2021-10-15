# Create emulation training data

This directory contains the configuration for creating physics emulation training data.

## Training data simulation config
These runs are initialized at the first of each calendar month from GFS data.  The initializiation data URL and argo command are in `run_single.sh`.  The complete set of runs are performed by `run_all.sh` where run configuration (`PROG_YAML`) and output url (`OUTPUT_URL`) are specified. 

The prognostic run configuration `fv3config.yaml` contains the information about variables and file to output. Default emulation training is set up to run Zhao-Carr microphysics and hybrid-EDMF turbulence (simple physics) and expects `state_after_dynamics.zarr`, `state_after_timestep.zarr`, and at least `physics_tendencies`.  Currently additional tendencies from physics components (e.g., microphysics) are output in `physics_component_tendencies.zarr`.

**Note**: The configuration file maps the parameter `initial_conditions` to `$IC_URL`, which is updated using `envsubst` in `run_single.sh`. This is used to initialize the simulations for every month in `run_all.sh`. 
