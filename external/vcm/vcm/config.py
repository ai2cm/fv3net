from typing import Mapping


def update_nested_dict(source: Mapping, update: Mapping) -> Mapping:
    """Recursively update a mapping with new values.

    Args:
        source: Mapping to be updated.
        update: Mapping whose key-value pairs will update those in source.
            Key-value pairs will be inserted for keys in update that do not exist
            in source.

    Returns:
        Recursively updated mapping.
    """
    if not isinstance(source, Mapping):
        raise TypeError(f"Expected source to be a mapping, got: {type(source)}")
    if not isinstance(update, Mapping):
        raise TypeError(f"Expected update to be a mapping, got: {type(update)}")
    for key in update:
        if key in source and isinstance(source[key], Mapping):
            update_nested_dict(source[key], update[key])
        else:
            source[key] = update[key]
    return source
