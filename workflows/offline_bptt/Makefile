DOCKER_SERVICE = offline_bptt
LABEL ?= nudge_3h

build:
	docker-compose build $(DOCKER_SERVICE)

dev:
	docker-compose run $(DOCKER_SERVICE) bash

train:
	python train.py $(LABEL)

evaluate:
	python evaluate.py $(LABEL)

clean:
	$(RM) *.npz *_packer-*.yml

.PHONY: build dev
