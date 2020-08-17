from typing import Mapping


def _update_nested_dict_once(source: Mapping, update: Mapping) -> Mapping:
    """Recursively update a mapping with new values.

    Args:
        source: Mapping to be updated.
        update: Mapping whose key-value pairs will update those in source.
            Key-value pairs will be inserted for keys in update that do not exist
            in source.

    Returns:
        Recursively updated mapping.
    """
    for key in update:
        if (
            key in source
            and isinstance(source[key], Mapping)
            and isinstance(update[key], Mapping)
        ):
            update_nested_dict(source[key], update[key])
        else:
            source[key] = update[key]
    return source


def update_nested_dict(*mappings) -> Mapping:
    """Recursive merge dictionaries updating from left to right.

    For example, the rightmost mapping will override the proceeding ones. """
    out, rest = mappings[0], mappings[1:]
    for mapping in rest:
        out = _update_nested_dict_once(out, mapping)
    return out
