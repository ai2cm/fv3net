import vcm
import fsspec
import json
import sys
import yaml


def flatten(seq):
    return [it for subseq in seq for it in subseq]


with open(sys.argv[1]) as f:
    args = yaml.safe_load(f)


fs = fsspec.filesystem("gs")
url = args.pop("url")
urls = sorted(fs.ls(url))
steps = [vcm.parse_timestep_str_from_path(url) for url in urls]
spinup = args.pop("spinup", steps[0])
include_one_step = args.pop("force_include_one_step", [])
steps = vcm.remove_spinup_period(steps, spinup)
splits = vcm.train_test_split_sample(steps, **args)

all_steps = sorted(set(flatten(flatten(splits.values()))))
all_steps += include_one_step
data = {"one_step": list(all_steps), "train_and_test": splits}

print(json.dumps(data))
