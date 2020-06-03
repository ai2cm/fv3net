SHELL=/bin/bash

CONFIG_FILENAME=example.yml
OUTDIR=outdir
RUNNER_FILENAME=runner.py
TRIAL_FILENAME=trial.py
MOUNT_LOCATION=/run
FV3KUBE_DIR=../../external/fv3kube
BASE_DIR=$(CWD)

ENV_NAME=tuning
PY_VERSION=3.6
REQUIREMENTS_FILENAME=requirements.txt

all: clean test

env:
	conda create -y -n $(ENV_NAME) python=$(PY_VERSION)
	source activate $(ENV_NAME) && conda install -y mongodb && pip install -r $(REQUIREMENTS_FILENAME)


clean:
	$(RM) -r $(OUTDIR)
	mkdir $(OUTDIR)
	touch $(OUTDIR)/.gitkeep

build:
	docker build . -f Dockerfile  -t $(IMAGE_NAME)

test:
	conda run -n $(ENV_NAME) python3 $(RUNNER_FILENAME) $(CONFIG_FILENAME) $(OUTDIR)

test-docker:
	docker run -v $(PWD)/$(CONFIG_FILENAME):$(MOUNT_LOCATION)/$(CONFIG_FILENAME) \
		-v $(PWD)/$(RUNNER_FILENAME):$(MOUNT_LOCATION)/$(RUNNER_FILENAME) \
		-v $(PWD)/$(TRIAL_FILENAME):$(MOUNT_LOCATION)/$(TRIAL_FILENAME) \
		-v $(PWD)/$(OUTDIR):$(MOUNT_LOCATION)/$(OUTDIR) \
		$(IMAGE_NAME) bash -c "cd $(MOUNT_LOCATION) && python3 $(RUNNER_FILENAME) $(CONFIG_FILENAME) $(OUTDIR)"
