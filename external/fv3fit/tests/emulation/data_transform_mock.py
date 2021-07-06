from toolz.functoolz import curry


@curry
def dymmy_transform(dataset):
    return dataset


@curry
def dummy_transform_w_kwarg(dataset, extra_kwarg=None):
    return dataset


@curry
def dummy_transform_w_leading_arg(leading_arg, dataset):
    return dataset
