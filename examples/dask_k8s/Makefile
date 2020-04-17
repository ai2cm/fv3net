IMAGE = dask-test
build:
	docker build -t $(IMAGE) .

submit:
	kubectl apply -f job.yaml -f dask.yaml

delete:
	kubectl delete -f job.yaml

dev:
	kubectl run temp -ti --image=$(IMAGE) -- pwd
	kubectl delete deployment temp

