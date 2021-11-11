UNIQUE = $(shell openssl rand --hex 6)
TAG ?= prognostic-experiment-$(UNIQUE)
TF_MODEL ?= gs://vcm-ml-experiments/2021-10-14-microphsyics-emulation-paper/models/all-tends-limited/all-tends-limited-dense/model.tf
CONFIG ?= configs/default.yaml
BASELINE_CONFIG = configs/baseline.yaml

baseline: TAG=test-baseline-1
baseline:
	argo submit prog-run-and-eval.yaml \
		-p tag=$(TAG) \
		-p config="$(file < $(BASELINE_CONFIG))"\
		-p tf_model=$(TF_MODEL) \
		-p on_off_flag="--offline" \
		-p do_piggy="false"

offline: TAG=offline-$(TAG)
offline:
	argo submit prog-run-and-eval.yaml \
		-p tag=$(TAG) \
		-p config="$(shell cat $(CONFIG))"\
		-p tf_model=$(TF_MODEL) \
		-p on_off_flag="--offline"

online: TAG=online-$(TAG)
online:
	argo submit prog-run-and-eval.yaml \
		-p tag=$(TAG) \
		-p config="$(shell cat $(CONFIG))"\
		-p tf_model=$(TF_MODEL) \
		-p on_off_flag="--online"


