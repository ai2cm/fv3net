IMAGE_NAME=us.gcr.io/vcm-ml/fv3net-tuning
CONFIG_FILENAME=example.yml
OUTDIR=outdir
RUNNER_FILENAME=runner.py
TRIAL_FILENAME=trial.py
MOUNT_LOCATION=/run

all: build test-docker

build:
	docker build . -f Dockerfile  -t $(IMAGE_NAME)

test:
	python3 $(RUNNER_FILENAME) $(CONFIG_FILENAME) $(OUTDIR)

test-docker:
	docker run -v $(PWD)/$(CONFIG_FILENAME):$(MOUNT_LOCATION)/$(CONFIG_FILENAME) \
		-v $(PWD)/$(RUNNER_FILENAME):$(MOUNT_LOCATION)/$(RUNNER_FILENAME) \
		-v $(PWD)/$(TRIAL_FILENAME):$(MOUNT_LOCATION)/$(TRIAL_FILENAME) \
		-v $(PWD)/$(OUTDIR):$(MOUNT_LOCATION)/$(OUTDIR) \
		$(IMAGE_NAME) bash -c "cd $(MOUNT_LOCATION) && python3 $(RUNNER_FILENAME) $(CONFIG_FILENAME) $(OUTDIR)"
