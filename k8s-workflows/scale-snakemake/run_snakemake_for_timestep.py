from typing import List, Any, Dict
import subprocess
import yaml
import os
import tempfile
import contextlib


@contextlib.contextmanager
def tempfile_context_manager():
    file = tempfile.mktemp()
    yield file
    os.unlink(file)


def run_snakemake_with_config(config: Dict):
    with tempfile_context_manager() as path:
        with open(path, "w") as f:
            yaml.dump(config, f)

        subprocess.check_call(['snakemake', '--configfile', path])


if __name__ == '__main__':
    import sys
    grids = ['C48']
    timesteps = sys.argv[1:]
    config = {'timesteps': timesteps}
    run_snakemake_with_config(config)
