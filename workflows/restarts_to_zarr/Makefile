IMAGE = us.gcr.io/vcm-ml/restarts_to_zarr
build:
	docker build -f Dockerfile  -t $(IMAGE) ../../

submit:
	kubectl apply -f job.yaml -f dask.yaml

delete:
	kubectl delete -f job.yaml
	kubectl delete pods -ldask-job=dask
push:
	docker push $(IMAGE)

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
