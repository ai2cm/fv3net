IMAGE=us.gcr.io/vcm-ml/prognostic-run-orchestration
KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
LOCAL_DIR_ARGS = -w /code -v $(shell pwd):/code

RUN_ARGS = --rm $(KEY_ARGS) $(LOCAL_DIR_ARGS) $(IMAGE)
RUN_INTERACTIVE = docker run -ti $(RUN_ARGS)
RUN ?= docker run $(RUN_ARGS)
SKLEARN_MODEL = gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl
FV3CONFIG = fv3config.yml
FV3NET_VERSION ?=2020-01-23-prognostic-rf

all: sklearn_run

fv3net-0.1.0-py3-none-any.whl:
	pip wheel --no-deps git+ssh://git@github.com/VulcanClimateModeling/fv3net.git@$(FV3NET_VERSION)

build: fv3net-0.1.0-py3-none-any.whl
	docker build . -t $(IMAGE)

fv3net-local:
	pip wheel --no-deps ../../.

vcm-local:
	pip wheel --no-deps ../../external/vcm

build_local: fv3net-local vcm-local
	docker build . -t $(IMAGE)

dev:
	 $(RUN_INTERACTIVE) bash

test_run_sklearn: state.pkl
	$(RUN) python3 -m sklearn_interface state.pkl $(SKLEARN_MODEL)

state.pkl:
	fv3run --dockerimage test-image --runfile save_state_runfile.py $(FV3CONFIG) save_state/
	cp save_state/state.pkl .

sklearn_run: #rundir
	fv3run --dockerimage us.gcr.io/vcm-ml/prognostic-run-orchestration --runfile sklearn_runfile.py $(FV3CONFIG) ../../scratch/rundir

clean:
	rm -rf net_precip net_heating/ PW
