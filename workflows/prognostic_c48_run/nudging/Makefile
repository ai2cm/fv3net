IMG_NAME?=us.gcr.io/vcm-ml/fv3gfs-python
IMG_VERSION?=v0.4.3
OUTDIR?=outdir
FV3RUN_FLAGS?=

BASE_CONFIG_FILE=./nudging_config.yaml
CONFIG_FILE=./fv3config.yml
RUNFILE=./runfile.py
K8S_JOB_YAML=./job.yaml
BASE_IMG=$(IMG_NAME):$(IMG_VERSION)
PREPARE_CONFIG_PY=prepare_config.py

all: run

run: $(CONFIG_FILE)
	fv3run $(CONFIG_FILE) $(OUTDIR) --dockerimage $(BASE_IMG) --runfile $(RUNFILE) $(FV3RUN_FLAGS)

$(CONFIG_FILE): $(BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) $(BASE_CONFIG_FILE) > $(CONFIG_FILE)

clean:
	$(RM) $(CONFIG_FILE) $(K8S_JOB_YAML)

.PHONY:clean run run_kubernetes
