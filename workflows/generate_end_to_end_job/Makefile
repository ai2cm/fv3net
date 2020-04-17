
timesteps.json:
	python print_samples.py > $@

one_step.json: timesteps.json
	jq .one_step timesteps.json > $@

training_times.json: timesteps.json
	jq .splits timesteps.json > $@
