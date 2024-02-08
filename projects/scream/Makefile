IMAGE_NAME = us.gcr.io/vcm-ml/prognostic_scream_run
IMAGE_TAG ?= cuda-gpu-test
IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)


enter_scream:
	docker run -it --rm \
		-w /storage/working \
		-v /home/andrep/repos/fv3net/docker/prognostic_scream_run/scream:/src/E3SM \
		-v ./inputdata:/storage/inputdata \
		-v ./build_scream:/src/build_scream \
		-v ./working:/storage/working \
		--mount source=bash_history,target=/root/.bash_history \
		-e HISTFILE=/root/.bash_history/history \
		$(IMAGE) bash

sleep:
	argo submit sleep.yaml

job: delete_sleep
	argo submit argo.yaml

delete_sleep:
	argo list --running -o name | grep sleep | xargs argo delete

running:
	argo list --running --prefix scream

all_jobs:
	argo list --prefix scream

clear_jobs:
	argo list --prefix scream -o name | xargs argo delete

push_scream:
	VERSION=cuda-gpu-test $(MAKE) -C ../.. push_image_prognostic_scream_run