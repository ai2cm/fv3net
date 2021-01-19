RUNDIR_CONFIG ?= ./configs/fv3config-1day-prognostic.yaml
IC_DIR ?= gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/
IC_TIMESTEP ?= 20160801.001500

RUNDIR ?= $(shell pwd)/rundir/
FV3NET_DIR ?= $(shell pwd)/../../
IMAGE ?= prognostic_run:emulation

PROGNOSTIC_DUMP ?= $(RUNDIR)/emulation_prognostic
KERAS_MODEL ?= gs://vcm-ml-scratch/andrep/emulation-models/2021-01-01-width-128-4ep/keras_model/


.PHONY: create_rundir run_prognostic prog_dev

all: create_rundir create_training rechunk_training

create_rundir:
	python prepare_rundir.py $(RUNDIR_CONFIG) $(IC_DIR) $(IC_TIMESTEP) $(RUNDIR)

run_prognostic:
	cp run_base.sh $(RUNDIR)/run.sh
	docker run  \
		-v $(RUNDIR):/rundir \
		-v $(PROGNOSTIC_DUMP):/state_out \
		-w /rundir \
		-e STATE_DUMP_PATH=/state_out \
		-e INPUT_NML_PATH=/rundir/input.nml \
		-e VARIABLE_METADATA_YAML=/tmp/emulation_slim/turb_parameter_metadata.yaml \
		-e TKE_EMU_MODEL=$(KERAS_MODEL) \
		-t $(IMAGE) \
		sh run.sh

prog_dev:
	cp run_base.sh $(RUNDIR)/run.sh
	docker run  \
		-v $(RUNDIR):/rundir \
		-v $(PROGNOSTIC_DUMP):/state_out \
		-v $(FV3NET_DIR):/fv3net \
		-w /rundir \
		-e STATE_DUMP_PATH=/state_out \
		-e INPUT_NML_PATH=/rundir/input.nml \
		-e VARIABLE_METADATA_YAML=/tmp/emulation_slim/turb_parameter_metadata.yaml \
		-e TKE_EMU_MODEL=$(KERAS_MODEL) \
		-it $(IMAGE) \
		bash





