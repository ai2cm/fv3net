## Building the prognostic run environment on Gaea

To build a prognostic run environment on Gaea from scratch, start by cloning the fv3net repository and initializing the submodules:
```
$ git clone https://github.com/ai2cm/fv3net.git
$ cd fv3net
$ git submodule update --init --recursive
```
Then run the `.environment-scripts/setup_environment.sh` script in a batch job (it will take a long time to complete).  It installs all of the compiled and Python dependencies of the prognostic run.  This includes NCEPlibs, ESMF, FMS, the fortran model, the Python requirements, the Python wrapped model, and the fv3net packages.  It does so using the system native Intel compilers.  The script takes a number of arguments.  An example call might look like:
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
Here `$CLONE_PREFIX` is the path to store the repositories downloaded during the process of installation; `$INSTALL_PREFIX` is the location to install non-local packages; `$FV3NET_DIR` is the absolute path to the location of the fv3net repository; `$CALLPYFORT` denotes whether `call_py_fort` will be installed (currently the only option here is `""` to denote that it will not be installed on Gaea); and finally `$CONDA_ENV` is the name of the conda environment you would like to create (it must not exist already).

## Activating the prognostic run environment

Once the prognostic run environment is installed, to activate it source the `.environment-scripts/activate_environment.sh` script.  This will activate the specified conda environment containing all the installed Python packages, and will also update the `LD_LIBRARY_PATH` appropriately to point to the ESMF and FMS shared libraries.  An example call might look like this:
```
$ source .environment-scripts/activate_environment.sh \
    gaea \
    $FV3NET_DIR \
    $INSTALL_PREFIX \
    $CONDA_ENV
```
where the variables have the same meaning as they did in the install script.  Generally you would call the activation script at the start of a batch job to run a simulation.
