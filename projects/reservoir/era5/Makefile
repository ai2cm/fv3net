.ONESHELL:

SHELL = /bin/bash
VERSION ?= v3-oct2023
IMAGE ?= gcr.io/vcm-ml/dataflow-xbeam:$(VERSION)
DATAFLOW_ENV ?= dataflow

build_docker:
	docker build -t ${IMAGE} .

push_docker: build_docker
	docker push ${IMAGE}

enter:
	docker run -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it ${IMAGE}

test_local:
	docker run -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it ${IMAGE} -c "./test_local.sh"

deploy:
	docker run -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it ${IMAGE} -c "./run_dataflow.sh"

clean:
	rm -f *.nc
