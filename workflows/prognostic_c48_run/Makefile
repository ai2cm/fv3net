KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
TAG ?= $(shell git rev-parse HEAD)
REGISTRY ?= us.gcr.io/vcm-ml/prognostic_run

build:
	docker-compose build fv3
	docker tag ${REGISTRY}:latest ${REGISTRY}:${TAG}

push: build
	docker push ${REGISTRY}:${TAG}

dev:
	docker-compose run $(KEY_ARGS) fv3 bash

test:
	docker-compose run fv3 pytest

.PHONY: dev test
