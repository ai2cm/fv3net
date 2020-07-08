IMAGE = us.gcr.io/vcm-ml/post_process_run:$(shell git rev-parse HEAD)

build:
	docker build -t $(IMAGE) .

push: build
	docker push $(IMAGE)