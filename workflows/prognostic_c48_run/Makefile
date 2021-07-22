KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json

REGISTRY=us.gcr.io/vcm-ml/prognostic_run
TAG=$(shell git rev-parse HEAD)

build:
	docker-compose build fv3
	docker tag $(REGISTRY):latest $(REGISTRY):$(TAG)
	docker tag $(REGISTRY):latest $(REGISTRY):microphysics_emulation

dev:
	docker-compose run $(KEY_ARGS) fv3 bash

test:
	docker-compose run fv3 pytest
	
indocker_rebuild:
	cd /tmp/fortran-build/FV3 && \
    ./configure gnu_docker && \
    $(MAKE) -j 8 && \
    PREFIX=/usr/local $(MAKE) install

.PHONY: dev test
