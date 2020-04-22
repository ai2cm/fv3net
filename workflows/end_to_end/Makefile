
build:
	docker build ../../ -f $(shell pwd)/Dockerfile -t e2e

requirements.txt: poetry.lock
	poetry export -f requirements.txt -o requirements.txt
