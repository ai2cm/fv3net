IMAGE_NAME = us.gcr.io/vcm-ml/prognostic_scream_run
IMAGE_TAG ?= cuda-gpu-test
IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
LOCAL_SCREAM_INPUT ?= $(PSCRATCH)/inputdata
SCREAM_DIR ?= $(HOME)/repos/fv3net/docker/prognostic_scream_run/scream


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

build_local:
	cmake -DCMAKE_CXX_COMPILER=$(shell which mpicxx) -DCMAKE_Fortran_COMPILER=$(shell which mpifort) \
		-DCMAKE_C_COMPILER=$(shell which mpicc) -DCMAKE_BUILD_TYPE=Debug -D SCREAM_DYNAMICS_DYCORE=HOMME \
		-D SCREAM_FPE=TRUE -D SCREAM_PACK_SIZE=1 -D SCREAM_SMALL_PACK_SIZE=1 -D EKAT_DEFAULT_BFB=ON \
		-D SCREAM_INPUT_ROOT:PATH=$(LOCAL_SCREAM_INPUT) \
		-D PnetCDF_C_INCLUDE_DIR=/opt/cray/pe/parallel-netcdf/1.12.3.9/gnu/12.3/include \
		-D PnetCDF_C_LIBRARY=/opt/cray/pe/parallel-netcdf/1.12.3.9/gnu/12.3/lib \
		-C $(SCREAM_DIR)/components/eamxx/cmake/machine-files/docker-scream.cmake \
		$(SCREAM_DIR)/components/eamxx