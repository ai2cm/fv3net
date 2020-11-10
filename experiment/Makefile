
all: kustomize
	./kustomize build . | kubectl apply -f -
	./run.sh

test: kustomize
	./kustomize build . | kubectl apply -f -
	./run_test.sh


kustomize:
	../install_kustomize.sh 3.8.6
    
