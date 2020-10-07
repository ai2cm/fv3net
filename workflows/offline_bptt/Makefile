DOCKER_SERVICE = offline_bptt

build:
	docker-compose build $(DOCKER_SERVICE)

dev:
	docker-compose run $(DOCKER_SERVICE) bash

run:
	python train.py default

clean:
	$(RM) *.npz *_packer-*.yml

.PHONY: build dev
