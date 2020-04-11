VERSION ?= v0.1.0
IMAGE = us.gcr.io/vcm-ml/metrics:$(VERSION)

build:
	docker build . -t $(IMAGE)

push: build
	docker push $(IMAGE)
