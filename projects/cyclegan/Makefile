

train_c48_to_c384: deploy
	cd c48_to_c384; ./run.sh

train_0K_to_8K: deploy
	cd 0K_to_8K; ./run.sh

deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0
