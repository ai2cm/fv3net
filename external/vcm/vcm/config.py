from typing import Mapping


def update_nested_dict(source: Mapping, update: Mapping) -> Mapping:
    """Recursively update a mapping with new values.

    Args:
        source: Mapping to be updated.
        update: Mapping whose key-value pairs will update those in source_dict.
            Key-value pairs will be inserted for keys in update_dict that do not exist
            in source_dict.

    Returns:
        Recursively updated mapping.
    """
    for key in update:
        if key in source and isinstance(source[key], Mapping):
            update_nested_dict(source[key], update[key])
        else:
            source[key] = update[key]
    return source
