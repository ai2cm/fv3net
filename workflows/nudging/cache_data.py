import fv3config
import yaml
import tempfile

with open("fv3config.yml", "r") as f:
    config = yaml.safe_load(f)
with tempfile.TemporaryDirectory() as tmpdir:
    fv3config.write_run_directory(config, tmpdir)
