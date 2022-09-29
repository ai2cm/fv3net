# Environment scripts

This directory contains a number of scripts that are used for installing
components of the prognostic run environment.  The main entrypoint is the
`setup_environment.sh` script, which can be used to install some or all of the
dependencies on a particular platform.  Currently supported platforms are:

- `gnu_docker`
- `gaea`

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
