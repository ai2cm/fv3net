IMG_NAME=us.gcr.io/vcm-ml/fv3gfs-python
IMG_VERSION=v0.4.0-nudging-beta
REMOTE_ROOT?=gs://vcm-ml-data/mcgibbon-testing
RUNFILE=./runfile.py
REMOTE_RUNFILE=$(REMOTE_ROOT)/runfile.py
BASE_CONFIG_FILE=./fv3config_base.yml
CONFIG_FILE=./fv3config.yml
REMOTE_CONFIG_FILE=$(REMOTE_ROOT)/config/fv3config.yml
OUTDIR?=outdir
REMOTE_OUTDIR=$(REMOTE_ROOT)/outdir
FV3RUN_FLAGS?=
KUBE_YML?=kube.yml

BASE_IMG=$(IMG_NAME):$(IMG_VERSION)
SUBMIT_JOB_PY=submit_job.py

all: run

run: $(CONFIG_FILE)
	fv3run $(CONFIG_FILE) $(REMOTE_OUTDIR) --dockerimage $(BASE_IMG) --runfile $(RUNFILE) $(FV3RUN_FLAGS)

run_kubernetes: $(CONFIG_FILE)
	gsutil cp $(CONFIG_FILE) $(REMOTE_CONFIG_FILE)
	gsutil cp $(RUNFILE) $(REMOTE_RUNFILE)
	python3 $(SUBMIT_JOB_PY) $(REMOTE_CONFIG_FILE) $(REMOTE_RUNFILE) $(REMOTE_OUTDIR) $(BASE_IMG) --detach

$(CONFIG_FILE): prepare_config.py $(BASE_CONFIG_FILE)
	python3 prepare_config.py $(BASE_CONFIG_FILE) $(CONFIG_FILE)

clean:
	$(RM) $(CONFIG_FILE)

.PHONY:clean run run_kubernetes
