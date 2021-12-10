# coding: utf-8
import wandb
import sys
import yaml


run_id = sys.argv[1]

api = wandb.Api()
run = api.run(run_id)
config = run.config
yaml.safe_dump(config, sys.stdout)
