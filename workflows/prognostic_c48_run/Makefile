IMAGE = us.gcr.io/vcm-ml/prognostic_run:latest
KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
LOCAL_DIR_ARGS = -w /code -v $(shell pwd):/code

RUN_ARGS = --rm $(KEY_ARGS) $(LOCAL_DIR_ARGS) $(IMAGE)
RUN_INTERACTIVE = docker run -ti $(RUN_ARGS)
RUN ?= docker run $(RUN_ARGS)
SKLEARN_MODEL = gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl
FV3CONFIG = gs://vcm-ml-data/end-to-end-experiments/2020-02-26-physics-off/annak-prognostic-physics-off-1773255e/prognostic_run_prognostic_yaml_adjust_prognostic_config.yml_ic_timestep_20160801.001500_docker_image_prognostic-run-orchestration/job_config/fv3config.yml

all: sklearn_run

dev:
	 $(RUN_INTERACTIVE) bash

sklearn_run: pull_image
	fv3run --dockerimage $(IMAGE) --runfile sklearn_runfile.py $(FV3CONFIG) rundir

pull_image:
	docker pull $(IMAGE)

clean:
	rm -rf net_precip net_heating/ PW

.PHONY: fv3net vcm build dev sklearn_run
