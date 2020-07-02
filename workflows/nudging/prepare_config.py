from datetime import datetime
import os
import sys
import yaml

import fv3kube
import vcm

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    reference_dir = config["nudging"]["restarts_path"]
    time = datetime(*config["namelist"]["coupler_nml"]["current_date"])
    label = vcm.encode_time(time)
    config = fv3kube.get_full_config(config, reference_dir, label)
    print(yaml.dump(config))
