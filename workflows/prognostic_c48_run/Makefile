IMAGE=test-image
KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
LOCAL_DIR_ARGS = -w /code -v $(shell pwd):/code

RUN_ARGS = --rm $(KEY_ARGS) $(LOCAL_DIR_ARGS) $(IMAGE)
RUN_INTERACTIVE = docker run -ti $(RUN_ARGS)
RUN ?= docker run $(RUN_ARGS)
MPIRUN = $(RUN_INTERACTIVE) mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none


build:
	pip wheel --no-deps git+ssh://git@github.com/VulcanClimateModeling/fv3net.git@2020-01-23-prognostic-rf
	docker build . -t $(IMAGE)

rundir:
	$(RUN) python3 download_rundir.py

dev:
	 $(RUN_INTERACTIVE) bash

test_run_sklearn: state.pkl
	$(RUN) python3 run_sklearn.py

state.pkl: rundir
	 $(MPIRUN) python3 save_state_runfile.py

sklearn_run: rundir
	$(MPIRUN) python3 sklearn_runfile.py

clean:
	rm -rf net_precip net_heating/ PW
