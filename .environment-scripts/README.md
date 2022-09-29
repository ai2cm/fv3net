# Environment scripts

This directory contains a number of scripts that are used for installing
components of the prognostic run environment.  The main entrypoint is the
`setup_environment.sh` script, which can be used to install some or all of the
dependencies on a particular platform.  As an illustration, to build a
prognostic run environment on Gaea from scratch, start by cloning the fv3net
repository and initializing the submodules:
```
$ git clone https://github.com/ai2cm/fv3net.git
$ cd fv3net
$ git submodule update --init --recursive
```
Then run the `.environment-scripts/setup_environment.sh` script in a batch job
(it will take a long time to complete).  It installs all of the compiled and
Python dependencies of the prognostic run.  This includes NCEPlibs, ESMF, FMS,
the fortran model, the Python requirements, the Python wrapped model, and the
fv3net packages.  It does so using the system native Intel compilers.  The
script takes a number of arguments.  An example call might look like:
```
$ sbatch --export=NONE bash .environment-scripts/setup_environment.sh \
    all \
    gaea \
    $CLONE_PREFIX \
    $INSTALL_PREFIX \
    $FV3NET_DIR \
    $CALLPYFORT \
    $CONDA_ENV
```
Here `$CLONE_PREFIX` is the path to store the repositories downloaded during the
process of installation; `$INSTALL_PREFIX` is the location to install non-local
packages; `$FV3NET_DIR` is the absolute path to the location of the fv3net
repository; `$CALLPYFORT` denotes whether `call_py_fort` will be installed
(currently the only option on Gaea is `""` to denote that it will not be
installed, but it is supported in Docker); and finally `$CONDA_ENV` is the name
of the conda environment you would like to create (it must not exist already).

## Activating the prognostic run environment

Once the prognostic run environment is installed, to activate it source the
`.environment-scripts/activate_environment.sh` script.  This will activate the
specified conda environment containing all the installed Python packages, and
will also update the `LD_LIBRARY_PATH` appropriately to point to the ESMF and
FMS shared libraries.  An example call might look like this:
```
$ source .environment-scripts/activate_environment.sh \
    gaea \
    $FV3NET_DIR \
    $INSTALL_PREFIX \
    $CONDA_ENV
```
where the variables have the same meaning as they did in the install script.
Generally you would call the activation script at the start of a batch job to
run a simulation.

## Currently supported platforms

Currently supported platforms are:

- `gnu_docker`
- `gaea` (call_py_fort integration is not supported yet)

## Adding support for new platforms

To add support for a new platform, the following steps must be taken:

1. One must define the following scripts in a subdirectory with the platform's
   name:
   - `<platform>/install_base_software.sh` -- this is a bit of a catch-all
     script that installs platform-specific software requirements.  In Docker
     this involves a fairly large number of dependencies, but on HPC platforms
     it typically only involves creating a base Python environment with Python
     version 3.8.10, and minimal initial dependencies (just `pip` and
     `pip-tools`).  
     - If the environment is based on a `conda` environment, the particular
       procedure for activating a conda environment on the platform must be
       defined in a `<platform>/activate_conda_environment.sh` script.
   - `<platform>/base_configuration_variables.sh` and
     `<platform>/development_configuration_variables.sh` -- these two scripts
     have similar functions but are split to be most compatible with our Docker
     caching scheme.  They define a number of platform-specific values for
     environment variables and script arguments used for installing various
     pieces of software. These variables are passed to steps of the
     `setup_base_environment.sh` and `setup_development_environment.sh` scripts,
     respectively.
2. A `configure.fv3.<platform>` file must be defined in the `FV3/conf` directory
   of the fv3gfs-fortran repository containing configuration information for
   building the fortran model and Python wrapper on the particular platform.
   Optionally, a `modules.fv3.<platform>` file can be created, which contains
   modules that will be loaded at the start of the environment building process.
