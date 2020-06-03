import collections
import os
import yaml


def load_config(config_dir):
    config = {}
    for filename in os.listdir(config_dir):
        if filename[-4:] == ".yml":
            base_name = filename[:-4]
        elif filename[-5:] == ".yaml":
            base_name = filename[:-5]
        else:
            base_name = None
        if base_name is not None:
            data = load_single_or_multiple_documents(os.path.join(config_dir, filename))
            config[base_name] = ConfigFile(filename, data)
    return config


def load_single_or_multiple_documents(yaml_filename):
    try:
        with open(yaml_filename, "r") as f:
            result = yaml.safe_load(f)
    except yaml.composer.ComposerError:
        with open(yaml_filename, "r") as f:
            result = list(yaml.safe_load_all(f))
    return result


def dump_single_or_multiple_documents(data, stream):
    if isinstance(data, collections.Sequence):
        yaml.safe_dump_all(data, stream)
    else:
        yaml.safe_dump(data, stream)


def write_config(config, config_dir):
    for name, config_file in config.items():
        filename = os.path.join(config_dir, config_file.filename)
        with open(filename, "w") as f:
            dump_single_or_multiple_documents(config_file.config, f)


class ConfigFile:
    def __init__(self, filename, config):
        self.filename = filename
        self.config = config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value
