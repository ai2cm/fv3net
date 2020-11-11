DOCKER_SERVICE = offline_bptt
LABEL ?= nudge_1day

build:
	docker-compose build $(DOCKER_SERVICE)

dev:
	docker-compose run $(DOCKER_SERVICE) bash

train:
	python train.py $(LABEL)

autoencoder:
	python autoencoder.py $(LABEL)

evaluate:
	python evaluate.py $(LABEL)

explore:
	python data_explore.py

block_explore:
	python block_explore.py $(LABEL)

clean:
	$(RM) *.npz *_packer-*.yml

.PHONY: build dev
