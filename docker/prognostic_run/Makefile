IMAGE = us.gcr.io/vcm-ml/prognostic_run:v0.1.0

all: build

fv3net:
	pip wheel --no-deps ../

vcm:
	pip wheel --no-deps ../external/vcm

build: fv3net vcm
	docker build -f Dockerfile ../../ -t $(IMAGE)

push: build
	docker push $(IMAGE)

.PHONY: fv3net vcm build dev sklearn_run
