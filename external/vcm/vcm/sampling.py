from typing import Any, Sequence
from toolz import sliding_window
import random


def sample(seq, n_samples, window=2):
    windows = list(sliding_window(window, seq))
    random.shuffle(windows)
    return windows[:n_samples]


def train_test_split_sample(seq, boundary, train_samples, test_samples):
    """
    >>> import pprint
    >>> timesteps = [
    ...     "20160101.000000",
    ...     "20160102.000000",
    ...     "20160103.000015",
    ...     "20160104.000000",
    ...     "20160201.000000",
    ...     "20160301.000000",
    ...     "20160401.000000",
    ...     "20160502.000000",
    ... ]
    ...
    ... timesteps = sorted(timesteps)
    >>> splits = train_test_split_sample(
        timesteps, "20160104.000000", train_samples=2, test_samples=2)
    >>> pprint.pprint(splits)
    {'test': [('20160401.000000', '20160502.000000'),
            ('20160104.000000', '20160201.000000')],
    'train': [('20160101.000000', '20160102.000000'),
            ('20160102.000000', '20160103.000015')]}
    """

    train = filter(lambda t: t < boundary, seq)
    test = filter(lambda t: t >= boundary, seq)

    train_steps = sample(train, train_samples)
    test_steps = sample(test, test_samples)

    return {"train": train_steps, "test": test_steps}
