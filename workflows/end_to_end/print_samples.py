"""
>>> max(steps)
'vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/20160910.000000/'
>>> min(steps)
'vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/20160801.001500/'
"""
import vcm
import fsspec
import json


url = "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48"

fs = fsspec.filesystem('gs')
urls = fs.ls(url)
steps = [vcm.parse_timestep_str_from_path(url) for url in urls]

splits = vcm.train_test_split_sample(steps, boundary="20160901.000000", train_samples=128, test_samples=128)
print(json.dumps(splits))
