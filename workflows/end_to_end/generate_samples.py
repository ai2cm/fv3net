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
splits = vcm.train_test_split_sample(
    steps,
    **args
)

all_steps = sorted(set(flatten(flatten(splits.values()))))
data = {"one_step": list(all_steps), "train": splits}

print(json.dumps(data))
