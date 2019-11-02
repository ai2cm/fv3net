all:
	python list_timesteps_to_run.py | bash scale_snakemake_argo.sh > _jobs.yml
	argo submit _jobs.yml
