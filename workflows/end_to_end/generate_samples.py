"""generate_samples.py

Output a dict describing the timesteps required for one-step jobs
and the split between training and testing data.

Takes a yaml as an argument, which has the following arguments:
    url: path to directory containing all available timesteps of initial conditions
    train_samples: number of samples to use for training
    test_samples: number of samples to use for testing
    boundary: timestamp splitting training and testing sets (before and after,
        respectively)
    spinup (optional): all times before this will be excluded from training/testing
        sets. By default, all timestamps at url can be used.
    force_include_one_step (optional): list of timestamps which will be required
        to be run for the one-step jobs. Does not affect training/testing samples.
        Defaults to an empty list.
    seed (optional): seed for random shuffling of training/testing samples
"""
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
steps = list(filter(lambda t: spinup <= t, steps))
splits = vcm.train_test_split_sample(steps, **args)

all_steps = sorted(set(flatten(flatten(splits.values()))))
all_steps += include_one_step
data = {"one_step": list(all_steps), "train_and_test": splits}

print(json.dumps(data))
