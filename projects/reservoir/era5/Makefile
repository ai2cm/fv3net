.ONESHELL:

SHELL = /bin/bash
VERSION ?= v2
IMAGE ?= gcr.io/vcm-ml/dataflow-xbeam:$(VERSION)
DATAFLOW_ENV ?= dataflow

build_docker:
	docker build -t ${IMAGE} .

push_docker: build_docker
	docker push ${IMAGE}

enter:
	docker run -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it ${IMAGE}

# A single day template with all target variables (no time coords)
# Used to stitch together a template to create the full daily zarr dataset
download_daily_template:
	gsutil cp gs://vcm-ml-archive/andrep/era5_rerid/single_day_template.nc .

test_local: download_daily_template
	docker run -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it ${IMAGE} -c "./test_local.sh"

deploy: download_daily_template
	conda activate ${DATAFLOW_ENV}
	./run_dataflow.sh

dataflow_conda_env:
	conda create -n ${DATAFLOW_ENV} python=3.10
	source activate ${DATAFLOW_ENV}
	pip install -r requirements.txt