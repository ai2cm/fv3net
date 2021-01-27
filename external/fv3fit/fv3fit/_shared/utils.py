from typing import List, Union


def parse_data_path(data_path: Union[List, str]):
    # allows the data path to be provided as a sequence of urls,
    # which is useful for hybrid training data
    if isinstance(data_path, List) and len(data_path) == 1:
        return data_path[0]
    else:
        return data_path
