IMAGE_NAME=us.gcr.io/vcm-ml/fv3net-tuning
CONFIG_FILENAME=example.yml
OUTDIR=outdir
RUNNER_FILENAME=runner.py
TRIAL_FILENAME=trial.py
MOUNT_LOCATION=/run
FV3KUBE_DIR=../../external/fv3kube

ENV_NAME=tuning
PY_VERSION=3.6
REQUIREMENTS_FILENAME=requirements.txt

all: build test-docker

env:
	conda create -n $(ENV_NAME) python=$(PY_VERSION)
	conda activate $(ENV_NAME)
	pip install -e $(FV3KUBE_DIR) && pip install $(REQUIREMENTS_FILENAME)

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
