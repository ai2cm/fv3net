# CycleGAN reference simulations

This directory contains configuration information and an example runscript for
the reference simulations used in McGibbon et al. (2023).  These simulations
were completed using the following versions of
[fv3net](https://github.com/ai2cm/fv3net) and
[fv3gfs-fortran](https://github.com/ai2cm/fv3gfs-fortran) on various partitions
of GFDL's Gaea computer.

- fv3net:
  `1d0967c5c7f7c347b25ecf7a2ce9ebd363e848ea`
- fv3gfs-fortran:
  `0a446035cc4730b9319a6925a438a22063117c7e`

The fortran executable and Python environment used to run these simulations can
be built on Gaea following the instructions
[here](https://github.com/ai2cm/fv3net/tree/master/.environment-scripts#building-an-environment-on-gaea).

## Configuration files

Configuration YAML files can be found in the `configurations` directory.  These
are defined for each resolution, C48 or C384, and climate (-2 K, 0 K, +2 K, +4
K, or ramped).  These YAML files are meant for use with
[fv3config](https://github.com/ai2cm/fv3config), which is a Python package
developed by our group for creating and manipulating run directories for FV3GFS
simulations.  The version of `fv3config` used (`0.9.0`), and the versions of all
other Python packages used in the scripts, are the same as those pinned in the
[`constraints.txt`
file](https://github.com/ai2cm/fv3net/blob/master/constraints.txt) in the fv3net
repository for the commit listed above.

## Example runscripts

An example runscript used for running on Gaea can be found in `run_fv3gfs.sh`.
This runscript uses the fv3config package to set up the run directory for each
segment of the simulation.  An example usage of it can be found below, e.g. for
running the C384 ramped simulation.

```bash
#!/bin/bash

ENVIRONMENT_ROOT=/ncrc/home1/Spencer.Clark/ai2/2023-04-06-CycleGAN-simulations-c4
PROJECT_ROOT=$ENVIRONMENT_ROOT/fv3net/projects/cyclegan_reference_simulations

SCRATCH_ROOT=$SCRATCH/$USER/2023-04-07-C384-CycleGAN-simulations
EXECUTABLE=$ENVIRONMENT_ROOT/install/bin/fv3.exe
SCRIPTS_DIR=$PROJECT_ROOT/scripts
CONDA_ENV=2023-04-06-fv3net-c4
PLATFORM=gaea
SEGMENTS=51

# Set SLURM environment variables for all submissions
export SBATCH_TIMELIMIT=06:30:00
export SLURM_NNODES=48

# Set configuration and name related variables
CONFIG_ROOT=$PROJECT_ROOT/configurations/C384/ramped

sbatch --nodes=$SLURM_NNODES --export=SBATCH_TIMELIMIT,SLURM_NNODES run_fv3gfs.sh \
       $SCRATCH_ROOT \
       C384-ramped \
       ${CONFIG_ROOT}/C384-ramped.yaml \
       $SEGMENTS \
       $EXECUTABLE \
       $CONDA_ENV \
       $SCRIPTS_DIR \
       $ENVIRONMENT_ROOT \
       $PLATFORM
```

Note that this is the pattern that we typically use in this project, since even
in the case that we use the `chgres_cube` tool to generate C384 initial
conditions from C48 restart files, this tool generates them in a form that makes
them look like initial conditions derived from GFS analysis.  The only time we
do a true branched simulation is when we run the C48 ramped simulation.  We
include a reference script for doing that in the form of `restart_fv3gfs.sh` as
well.

## Initial conditions

Initial conditions for the C48 simulations are derived from GFS analysis for
2016-01-01 00Z.  The surface temperature in the initial conditions is perturbed
by the climate-specific amount; one year is given for each of these simulations
to spin up.  Initial conditions for the C384 simulations are derived from
restart files from the first of each post-spin-up year of the C48 simulations in
each climate.  These initial conditions were derived using the `chgres_cube`
tool of the `UFS_UTILS` project using [the argo workflow defined in the fv3net
repository](https://github.com/ai2cm/fv3net/tree/master/workflows/argo#restart-files-to-nggps-initial-condition-workflow).
Note that for two select cases (year 2023 in the 0 K climate and year 2020 in
the +4 K climate) we needed to restart from restart files one month prior to
avoid a crash upon initialization.  The initialization crashes can be traced
back to the fact that sharp topographic features that exist at C384 resolution
do not exist at C48 resolution, and so sometimes the state interpolated to the
finer grid is not initially compatible with the fine resolution topography.

The initial conditions for the ramped simulations were derived from C48 restart
files from January 1st, 2017.  The C48 simulation was started directly using
these restart files, and the C384 simulation was started from an initial
condition constructed with the `chgres_cube` tool.

The C48 2016-01-01 00Z initial condition can be found in our cloud storage here:
```
gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016010100
```
and can be made available upon request.

## Ramped sea surface temperature forcing data

The ramped simulations require the definition of a netCDF file containing the
sea surface temperature on a latitude-longitude grid defined at monthly time
intervals.  We derived this forcing by loading in the climatological sea surface
temperature, defined for each month of the year, and repeating it for 51 months
(four years and three months).  We added a linearly increasing perturbation to
this time series starting at the start of the fourth month.  The evolution of
this perturbation can be found in the graph below.

![sst-offset-timeseries.png](sst-offset-timeseries.png?raw=true)

A script, `generate_ramped_sst.py`, that generates this ramped SST forcing file
is also included for reference.  See also
[here](https://github.com/ai2cm/fv3gfs-fortran/blob/cb106f2eb806e8c635d28d8b76ee8e80a0e20bc3/tests/pytest/prescribed_ssts.py#L3-L28)
for more information on how to generate an SST forcing file that can loaded by
FMS.

## Standard forcing data

We run FV3GFS using standard forcing data provided for the UFS weather model.
This consists of:

- Grid data: `gs://vcm-fv3config/data/grid_data/v1.0`
- Orographic forcing data: `gs://vcm-fv3config/data/orographic_data/v1.0`
- Surface forcing data: `gs://vcm-fv3config/data/base_forcing/v1.1`

Public versions of the grid and orographic forcing data for a given resolution
can be found
[here](https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/gfs.v16.3.7/fix/fix_fv3_gmted2010/),
and public versions of the surface forcing data can be found
[here](https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/gfs.v16.3.7/fix/fix_am/).
The specific files used can also be made available upon request.
