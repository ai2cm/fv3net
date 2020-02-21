IMG_NAME=us.gcr.io/vcm-ml/fv3gfs-python
IMG_VERSION=v0.3.1
RUNFILE=./runfile.py
BASE_CONFIG_FILE=./fv3config_base.yml
CONFIG_FILE=./fv3config.yml
OUTDIR=./outdir
TEMPORARY_CREDENTIALS=./temporary_gcs_credentials.json

BASE_IMG=$(IMG_NAME):$(IMG_VERSION)
NUDGING_IMG=$(IMG_NAME):$(IMG_VERSION)-nudging

all: build run

run: $(CONFIG_FILE)
	fv3run $(CONFIG_FILE) $(OUTDIR) --dockerimage $(NUDGING_IMG) --runfile $(RUNFILE)

build: $(CONFIG_FILE)
ifeq ($(GOOGLE_APPLICATION_CREDENTIALS),)
	$(error GOOGLE_APPLICATION_CREDENTIALS must be set to cache private data for image)
endif
ifeq ($(shell docker images -q $(BASE_IMG) 2> /dev/null),)
		docker pull $(BASE_IMG)
endif
	cp $(GOOGLE_APPLICATION_CREDENTIALS) $(TEMPORARY_CREDENTIALS)  # credentials have to be copied inside build context
	docker build -f Dockerfile --build-arg MODEL_IMAGE=$(BASE_IMG) \
		--build-arg GCS_CREDENTIALS=$(TEMPORARY_CREDENTIALS) \
		--target nudging -t $(NUDGING_IMG) . || (ret=$$?; rm -f $(TEMPORARY_CREDENTIALS) && exit $$ret)
	rm $(TEMPORARY_CREDENTIALS)

$(CONFIG_FILE): prepare_config.py $(BASE_CONFIG_FILE)
	python3 prepare_config.py $(BASE_CONFIG_FILE) $(CONFIG_FILE)

clean:
	$(RM) $(CONFIG_FILE)

.PHONY: build clean run
 