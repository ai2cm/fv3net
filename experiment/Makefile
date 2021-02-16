
N2FINE_TRAIN_TEST_DATA=gs://vcm-ml-experiments/2020-12-17-n2f-moisture-conservation/control/nudge-to-fine-run


run_winds_only_seed: deploy
	./run_random_seed.sh 2021-01-26-c3072-nn n2f $(N2FINE_TRAIN_TEST_DATA) winds

run_Tq_only_seed: deploy
	./run_random_seed.sh 2021-01-26-c3072-nn/corrected-learning-rate n2f $(N2FINE_TRAIN_TEST_DATA) tq

deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	../install_kustomize.sh 3.8.6

