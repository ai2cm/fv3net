import argparse
import numpy as np
import random


def batch_split(
        ds,
        num_batches
):


def train_test_split(
        ds,
        train_frac,
        test_frac,
        num_samples=None,
        random_seed=1234
):
    if (train_frac + test_frac > 1.):
        raise ValueError("Train + test fraction sums must be <= 1.")
    sample_indices = ds.sample.values
    total_num_datapoints = len(sample_indices)
    if not num_samples:
        num_samples = total_num_datapoints
    num_train = int(train_frac * total_num_datapoints)
    num_test = int(test_frac * total_num_datapoints)
    random.seed(random_seed)
    random.shuffle(sample_indices)
    ds_train = ds.isel(sample=sample_indices[:num_train])
    ds_test = ds.isel(sample=sample_indices[-num_test:])
    return ds_train, ds_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
