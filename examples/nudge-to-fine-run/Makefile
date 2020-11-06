
all: kustomize
	./kustomize build . | kubectl apply -f -
	./run.sh

kustomize:
	../../install_kustomize.sh 3.8.6

