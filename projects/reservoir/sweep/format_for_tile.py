import argparse

import yaml


def get_single_variable(data, keys):
    value = data
    for key in keys:
        value = value[key]
    return value


def set_single_variable(data, keys, new_value):
    value = data
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = new_value


def main(path, tile_number, variables):
    # Load the YAML file
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Loop through the mapping members and format the strings with the tile number
    for key_path in variables:
        keys = key_path.split(".")
        value = get_single_variable(data, keys)
        new_value = value.format(tile_number)
        set_single_variable(data, keys, new_value)

    # Print the updated YAML
    print(yaml.dump(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format YAML file for a specific tile number"
    )
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file to format")
    parser.add_argument(
        "tile_number", type=int, help="Tile number to format the YAML file for"
    )
    parser.add_argument(
        "variables",
        nargs="+",
        help="List of specific variables in the yaml to update with the tile number",
    )
    args = parser.parse_args()
    main(args.yaml_path, args.tile_number, args.variables)
