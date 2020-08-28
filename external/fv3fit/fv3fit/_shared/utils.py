import os
from typing import List


def parse_data_path(args):
    # allows the data path to be provided as a sequence of urls,
    # which is useful for hybrid training data
    if not args.no_train_subdir_append:
        data_path = [os.path.join(path, "train") for path in args.train_data_path]
    else:
        data_path = args.train_data_path
    if isinstance(args.train_data_path, List) and len(data_path) == 1:
        return data_path[0]
    else:
        return data_path
