from typing import List


def parse_data_path(args):
    # allows the data path to be provided as a sequence of urls,
    # which is useful for hybrid training data
    data_path = args.train_data_path
    if isinstance(args.train_data_path, List) and len(data_path) == 1:
        return data_path[0]
    else:
        return data_path
