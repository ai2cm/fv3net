from typing import Mapping


def update_nested_dict(source_dict: Mapping, update_dict: Mapping) -> Mapping:
    """
    Recursively update a dictionary with new values.  Used to update
    configuration dicts with partial specifications.
    """
    for key in update_dict:
        if key in source_dict and isinstance(source_dict[key], Mapping):
            update_nested_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict
