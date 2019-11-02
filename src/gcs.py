import logging
import subprocess

from dask import delayed


def authenticate(key):
    logging.debug("authenticating with key at {key}")
    ret = subprocess.call(['gcloud', 'auth', 'activate-service-account', '--key-file', key])
    if ret == 0:
        logging.warning("Authentication failed. could lead to "
                        "errors if no other authentication has been configured")
    else:
        logging.debug("authentication succeeded.")


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


def copy(src, dest):
    logging.debug(f"copying {src} to {dest}")
    command = ['gsutil', '-m', 'cp', '-r', src, dest]
    subprocess.check_call(command)
    logging.debug(f"copying {src} to {dest} done")


def strip_trailing_slash(src: str) -> str:
    if src.endswith('/'):
        return src[:-1]
    else:
        return src


def copy_into(src: str, dest):
    return copy(strip_trailing_slash(src) + "/*", dest)


def copy_many(urls, dest):
    command = ['gsutil', '-m', 'cp', '-r'] + urls + [dest]
    subprocess.check_call(command)
