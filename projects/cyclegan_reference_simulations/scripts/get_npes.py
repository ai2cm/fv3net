import argparse

import yaml


N_TILES = 6

parser = argparse.ArgumentParser()
parser.add_argument("config")
args, extra_args = parser.parse_known_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

layout_x, layout_y = config["namelist"]["fv_core_nml"]["layout"]
print(N_TILES * layout_x * layout_y)
