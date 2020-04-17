"""
>>> max(steps)
'vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/20160910.000000/'
>>> min(steps)
'vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/20160801.001500/'
"""
from kubernetes.client import V1ConfigMap
import vcm
import fsspec
import json
import sys

def flatten(seq):
    return [it 
    for subseq in seq
    for it in subseq
    ]

def jsonify(cm):
    from kubernetes.client import ApiClient
    api = ApiClient()
    return json.dumps(api.sanitize_for_serialization(cm))



url = "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48"

fs = fsspec.filesystem("gs")
urls = sorted(fs.ls(url))
steps = [vcm.parse_timestep_str_from_path(url) for url in urls]

splits = vcm.train_test_split_sample(
    steps, boundary="20160901.000000", train_samples=48, test_samples=48
)

all_steps = set(flatten(flatten(splits.values())))
# print(all_steps)

data = {
    "one_step": list(all_steps),
    "splits": splits['train']
}
json.dump(data, sys.stdout)
