submit: env
	bash -c 'source env/bin/activate && sh submit_job.sh'

submit_local: env
	bash -c 'source env/bin/activate && python pipeline.py'

env:
	python -m venv env
	bash -c 'source env/bin/activate && pip install -e .'


