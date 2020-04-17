import vcm
import fsspec
import json
import argparse


def flatten(seq):
    return [it for subseq in seq for it in subseq]


parser = argparse.ArgumentParser()
parser.add_argument("url", type=str)
parser.add_argument("boundary", type=str)
parser.add_argument("n_train", type=int)
parser.add_argument("n_test", type=int)
parser.add_argument("seed", type=str)

args = parser.parse_args()

fs = fsspec.filesystem("gs")
urls = sorted(fs.ls(args.url))
steps = [vcm.parse_timestep_str_from_path(url) for url in urls]
splits = vcm.train_test_split_sample(
    steps,
    boundary=args.boundary,
    train_samples=args.n_train,
    test_samples=args.n_test,
    seed=args.seed,
)

all_steps = sorted(set(flatten(flatten(splits.values()))))
data = {"one_step": list(all_steps), "train": splits}

print(json.dumps(data))
