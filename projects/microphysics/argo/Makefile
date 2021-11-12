UNIQUE = $(shell openssl rand --hex 4)

CONFIG ?= configs/default.yaml
TAG ?= prognostic-experiment-$(UNIQUE)
# BASELINE_TAG only used in online experiment for diag comparison
BASELINE_TAG ?= baseline
TF_MODEL ?= gs://vcm-ml-experiments/2021-10-14-microphsyics-emulation-paper/models/all-tends-limited/all-tends-limited-dense/model.tf
ON_OFF_FLAG ?= --offline
PIGGY_DIAG ?= true

submit:
	config_content=$$(cat $(CONFIG)) && \
	argo submit prog-run-and-eval.yaml \
		-p tag=$(TAG) \
		-p baseline_tag=$(BASELINE_TAG) \
		-p config="$$config_content"\
		-p tf_model=$(TF_MODEL) \
		-p on_off_flag="$(ON_OFF_FLAG)" \
		-p do_piggy="$(PIGGY_DIAG)" 


baseline: TAG=baseline
baseline: CONFIG=configs/baseline.yaml
baseline: PIGGY_DIAG=false
baseline: submit

offline: ON_OFF_FLAG=--offline
offline: TAG:=offline-$(TAG)
offline: submit

online: ON_OFF_FLAG=--online
online: TAG:=online-$(TAG)
online: submit


all_summer_models:
	run_all_summer_models.sh


