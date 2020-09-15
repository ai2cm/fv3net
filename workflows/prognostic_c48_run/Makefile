KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json

build:
	docker-compose build fv3

dev:
	docker-compose run $(KEY_ARGS) fv3 bash

test:
	docker-compose run fv3 pytest

.PHONY: dev test
