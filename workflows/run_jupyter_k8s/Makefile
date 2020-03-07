POD ?= $(shell kubectl get pods --selector 'app=jupyter-notebook' -o json | jq '.items[].metadata.name')
submit:
	kubectl apply -f jupyter.yaml

delete:
	kubectl delete -f jupyter.yaml

port-forward:
	kubectl port-forward $(POD) 9000:8888

logs:
	kubectl logs $(POD)

image:
	docker build . -t us.gcr.io/vcm-ml/noah-test-dask:latest

push: image
	docker push us.gcr.io/vcm-ml/noah-test-dask:latest
submit_job:
	kubectl apply -f daskjob.yaml

delete_job:
	kubectl delete -f daskjob.yaml
