INSTALL_PREFIX=$1

module rm PrgEnv-intel PrgEnv-cce PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-intel
module rm intel
module load intel/19.0.5.281
module load cray-netcdf
module load craype-hugepages4M
module unload cray-mpich
module load cray-mpich/7.7.11
module load cray-hdf5
module load cmake

# We don't use module activate python/3.9 since that pollutes our PATH
CONDA_PATH=/ncrc/sw/gaea-cle7/python/3.9/anaconda-base
CONDA_SETUP="$($CONDA_PATH/bin/conda shell.bash hook 2> /dev/null)"
eval "$CONDA_SETUP"

CONDA_PREFIX=$INSTALL_PREFIX/conda
CONDA_ENV=prognostic-run-clean-38
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs
conda deactivate

export FC=ftn
export CC=cc
export CXX=CC
export LD=ftn
export MPICC=cc  # Needed for building mpi4py
export TEMPLATE=site/intel.mk
export AVX_LEVEL=-xCORE-AVX2
export NETCDF_INCLUDE=$CRAY_NETCDF_PREFIX/include  # Needed for building ESMF

# https://stackoverflow.com/questions/70597896/check-if-conda-env-exists-and-create-if-not-in-bash
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*${CONDA_ENV}.*" ; then
    echo "Conda environment already exists.  Activating..."
    conda activate $CONDA_ENV
else
    echo "Conda environment doesn't exist.  Creating..."
    conda create -n $CONDA_ENV -c conda-forge python==3.8.10 pip pip-tools
    conda activate $CONDA_ENV
fi

PYTHON=`which python`
CONDA=`which conda`

echo "Compiler settings:"
echo "FC is     $FC"
echo "CC is     $CC"
echo "CXX is    $CXX"
echo "LD is     $LD"
echo "MPICC is  $MPICC"
echo
echo "Python settings:"
echo "conda is  $CONDA"
echo "python is $PYTHON"
echo
echo "Base enviroment packages:"
conda list
