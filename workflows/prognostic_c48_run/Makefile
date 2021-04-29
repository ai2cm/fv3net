KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json


# All physics emulation training create variables
curr_date = $(shell date +%Y-%m-%d)
target_tag = "allphys-emu-train-create-${curr_date}"
image_name = "us.gcr.io/vcm-ml/prognostic_run"

build:
	docker-compose build fv3

dev:
	docker-compose run $(KEY_ARGS) fv3 bash

test:
	docker-compose run fv3 pytest

build_push_train: build
	docker image tag ${image_name}:latest ${image_name}:${target_tag}
	docker push ${image_name}:${target_tag}

.PHONY: dev test
