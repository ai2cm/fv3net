IMAGE = us.gcr.io/vcm-ml/post_process_run:$(shell git rev-parse HEAD)

build_image:
	docker build -t $(IMAGE) .

push_image: build_image
	docker push $(IMAGE)