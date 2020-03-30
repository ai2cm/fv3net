IMAGE = dask
build:
	docker build -f Dockerfile  -t $(IMAGE) ../../

submit:
	kubectl apply -f job.yaml -f dask.yaml

delete:
	kubectl delete -f job.yaml
	kubectl delete pods -ldask-job=dask

dev:
	kubectl run temp -ti --image=$(IMAGE) -- pwd
	kubectl delete deployment temp

logs:
	kubectl logs job/dask-job
	kubectl logs -ldask-job=dask

redo:
	$(MAKE) build
	$(MAKE) delete 
	$(MAKE) submit
