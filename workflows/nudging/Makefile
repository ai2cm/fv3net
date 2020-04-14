IMG_NAME?=us.gcr.io/vcm-ml/fv3gfs-python
IMG_VERSION?=v0.4.0-nudging-beta
TIMESCALE_HOURS?=3
REMOTE_ROOT?=gs://vcm-ml-data/mcgibbon-testing
OUTDIR?=outdir
FV3RUN_FLAGS?=

RUN_LABEL=$(TIMESCALE_HOURS)h
BASE_CONFIG_FILE=./fv3config_base.yml
CONFIG_FILE=./fv3config-$(RUN_LABEL).yml
RUNFILE=./runfile.py
REMOTE_CONFIG_FILE=$(REMOTE_ROOT)/config/fv3config-$(RUN_LABEL).yml
REMOTE_RUNFILE=$(REMOTE_ROOT)/runfile.py
REMOTE_OUTDIR=$(REMOTE_ROOT)/outdir-$(RUN_LABEL)
BASE_IMG=$(IMG_NAME):$(IMG_VERSION)
SUBMIT_JOB_PY=submit_job.py
PREPARE_CONFIG_PY=prepare_config.py

all: run

run: $(CONFIG_FILE)
	fv3run $(CONFIG_FILE) $(OUTDIR) --dockerimage $(BASE_IMG) --runfile $(RUNFILE) $(FV3RUN_FLAGS)

run_kubernetes: $(SUBMIT_JOB_PY) $(CONFIG_FILE)
	gsutil cp $(CONFIG_FILE) $(REMOTE_CONFIG_FILE)
	gsutil cp $(RUNFILE) $(REMOTE_RUNFILE)
	python3 $(SUBMIT_JOB_PY) $(REMOTE_CONFIG_FILE) $(REMOTE_RUNFILE) $(REMOTE_OUTDIR) $(BASE_IMG) --detach

$(CONFIG_FILE): $(PREPARE_CONFIG_PY) $(BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) $(BASE_CONFIG_FILE) $(CONFIG_FILE) --timescale-hours $(TIMESCALE_HOURS)

clean:
	$(RM) $(CONFIG_FILE) $(RUNFILE)

.PHONY:clean run run_kubernetes
