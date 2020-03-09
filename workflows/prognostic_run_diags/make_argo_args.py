# coding: utf-8
import yaml
with open("input.yml") as f:
    rundirs = yaml.safe_load(f)
    
def name_output_bucket(rundir):
    return 'gs://vcm-ml-public' + rundir.lstrip('gs://vcm-ml-data/').rstrip('/') + '/index.html'
    
args = {'args': [{'rundir': rundir, 'output': name_output_bucket(rundir)} for rundir in rundirs]}
with open("args.yml","w") as f:
    yaml.dump(args, f)
    
