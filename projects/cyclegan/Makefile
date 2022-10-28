

train_c48_to_c384: deploy
	cd c48_to_c384; ./run.sh

deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0
