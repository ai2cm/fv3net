import tensorflow, fv3net, fv3fit  # noqa
import subprocess
import sys
import os


def test_gsutil_not_from_conda():
    gsutil_path = subprocess.check_output(["which", "gsutil"])
    conda_bin_dir = os.path.dirname(sys.executable)
    assert not gsutil_path.startswith(conda_bin_dir)


if __name__ == "__main__":
    test_gsutil_not_from_conda()
