python prepare_config.py --model_url \
gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-cd89364ac05d/trained_model \
--prog_config_yml config_24proc.yaml \
gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts  20160805.000000 \
>  config_24proc.full.yaml

write_run_directory config_24proc.full.yaml rundir
