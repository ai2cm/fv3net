import logging
import subprocess

from dask import delayed


@delayed
def upload_to_gcs(src, dest, save_op):
    logging.info("uploading %s to %s" % (src, dest))
    subprocess.check_call(['gsutil', '-q', 'cp',  src, dest])
    logging.info("uploading %s done" % dest)


def exists(url):
    proc = subprocess.call(['gsutil', 'ls', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc == 0


def list(pattern):
    files = subprocess.check_output(
        ['gsutil', 'ls', pattern]
    )
    return [arg.decode('UTF-8') for arg in files.split()]