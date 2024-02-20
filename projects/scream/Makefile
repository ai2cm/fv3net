IMAGE_NAME = us.gcr.io/vcm-ml/prognostic_scream_run
IMAGE_TAG ?= cuda-gpu-test
IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
LOCAL_SCREAM_INPUT ?= $(PSCRATCH)/inputdata
SCREAM_DIR ?= $(HOME)/repos/fv3net/docker/prognostic_scream_run/scream
NUM_PARALLEL_JOBS ?= 16


enter_scream:
	docker run -it --rm \
		-w /storage/working \
		-v /home/andrep/repos/fv3net/docker/prognostic_scream_run/scream:/src/E3SM \
		-v ./inputdata:/storage/inputdata \
		-v ./build_scream:/src/build_scream \
		-v ./working:/storage/working \
		--mount source=bash_history,target=/root/.bash_history \
		-e HISTFILE=/root/.bash_history/history \
		$(IMAGE) bash

sleep:
	argo submit sleep.yaml

job: delete_sleep
	argo submit argo.yaml

delete_sleep:
	argo list --running -o name | grep sleep | xargs argo delete

running:
	argo list --running --prefix scream

all_jobs:
	argo list --prefix scream

clear_jobs:
	argo list --prefix scream -o name | xargs argo delete

push_scream:
	VERSION=cuda-gpu-test $(MAKE) -C ../.. push_image_prognostic_scream_run

build_local_pm:
    . /usr/share/lmod/8.3.1/init/sh && \
	module unload cray-hdf5-parallel cray-netcdf-hdf5parallel 
		cray-parallel-netcdf cray-netcdf cray-hdf5 PrgEnv-gnu PrgEnv-intel PrgEnv-nvidia PrgEnv-cray \
		PrgEnv-aocc intel intel-oneapi nvidia aocc cudatoolkit climate-utils craype-accel-nvidia80 \
		craype-accel-host perftools-base perftools darshan && \
	module load PrgEnv-gnu/8.3.3 gcc/11.2.0 cray-libsci/23.02.1.1 craype-accel-host \
		craype/2.7.20 cray-mpich/8.1.25 cray-hdf5-parallel/1.12.2.3 cray-netcdf-hdf5parallel/4.9.0.3 \
		cray-parallel-netcdf/1.12.3.3 evp-patch && \
	export MPICH_ENV_DISPLAY=1 && \
	export MPICH_VERSION_DISPLAY=1 && \
	export OMP_STACKSIZE=128M && \
	export OMP_PROC_BIND=spread && \
	export OMP_PLACES=threads && \
	export HDF5_USE_FILE_LOCKING=FALSE && \
	export PERL5LIB=/global/cfs/cdirs/e3sm/perl/lib/perl5-only-switch && \
	export FI_CXI_RX_MATCH_MODE=software && \
	export MPICH_COLL_SYNC=MPI_Bcast && \
	export Albany_ROOT=/global/common/software/e3sm/mali_tpls/albany-e3sm-serial-release-gcc && \
	export Trilinos_ROOT=/global/common/software/e3sm/mali_tpls/trilinos-e3sm-serial-release-gcc && \
	export NETCDF_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.3/gnu/9.1 && \
	export PNETCDF_PATH=/opt/cray/pe/parallel-netcdf/1.12.3.3/gnu/9.1 && \
	export ADIOS2_ROOT=/global/cfs/cdirs/e3sm/3rdparty/adios2/2.9.1/cray-mpich-8.1.25/gcc-11.2.0 && \
	export BLA_VENDOR=Generic && \
	cmake --debug-output -C $(SCREAM_DIR)/components/eamxx/cmake/machine-files/docker-scream.cmake \
		-DNetCDF_Fortran_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.3/gnu/9.1 \
		-DNetCDF_C_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.3/gnu/9.1 \
		-DCMAKE_BUILD_TYPE=Debug \
		-DEKAT_DEFAULT_BFB=True \
		-DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.20/bin/CC \
		-DCMAKE_C_COMPILER=/opt/cray/pe/craype/2.7.20/bin/cc \
		-DCMAKE_Fortran_COMPILER=/opt/cray/pe/craype/2.7.20/bin/ftn \
		-DSCREAM_DYNAMICS_DYCORE=HOMME \
		-D SCREAM_INPUT_ROOT:PATH=$(LOCAL_SCREAM_INPUT) \
		$(SCREAM_DIR)/components/eamxx && \
	make -j$(NUM_PARALLEL_JOBS)
