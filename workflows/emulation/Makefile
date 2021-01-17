RUNDIR_CONFIG ?= ./configs/fv3config-20day-training.yaml
IC_DIR ?= gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/
IC_TIMESTEP ?= 20160801.001500

RUNDIR ?= $(shell pwd)/rundir/
IMAGE ?= prognostic_run:emu-create-train
STATE_DUMP ?= $(RUNDIR)/emulation_training
TMP_RECHUNK ?= /tmp/rechunked/


.PHONY: create_rundir create_training_data rechunk_training_data

all: create_rundir create_training rechunk_training

create_rundir:
	python prepare_rundir.py $(RUNDIR_CONFIG) $(IC_DIR) $(IC_TIMESTEP) $(RUNDIR)

##  Use emulation directory as working dir
## Add state dump as environment variable
create_training: 
	cp run_base.sh $(RUNDIR)/run.sh
	docker run  \
		-v $(RUNDIR):/rundir \
		-v $(STATE_DUMP):/state_out \
		-w /rundir \
		-e STATE_DUMP_PATH=/state_out \
		-e INPUT_NML_PATH=/rundir/input.nml \
		-t $(IMAGE) \
		sh run.sh

# Not elegant
rechunk_training:
	docker run \
		-v $(STATE_DUMP):/state_out \
		-v $(TMP_RECHUNK):/tmp/rechunk \
		-v $(shell pwd):/workdir \
		-w /workdir \
		-t $(IMAGE) \
		sh rechunk_zarr.sh





