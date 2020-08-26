# execute from docker
all: err

log err: rundir
	cd rundir && mpirun -np 6 \
	--oversubscribe \
	--allow-run-as-root \
	--mca btl_vader_single_copy_mechanism none \
	python $(shell pwd)/../sklearn_runfile.py > $(shell pwd)/log 2> $(shell pwd)/err

# execute from fv3net environment
rundir: fv3config.yaml
	write_run_directory fv3config.yaml $@
	cp fv3config.yaml $@/fv3config.yml

fv3config.yaml:
	python ../orchestrate_submit_job.py config.yaml output \
	--diag-table $(shell pwd)/../diag_table_prognostic \
	--model_url gs://vcm-ml-experiments/2020-08-06-hybrid-correct-cvcp/trained-model \
	> $@

clean:
	rm -rf fv3config.yaml rundir