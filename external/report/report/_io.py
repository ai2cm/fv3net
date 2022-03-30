import pathlib
import hashlib
import sys
from google.cloud import storage

BUCKET = "vcm-ml-public"
PROJECT = "vcm-ml"
PATH = "report"


def upload_html(path: str, contents: str):
    client = storage.Client(project=PROJECT)
    bucket = client.get_bucket(BUCKET)
    blob = bucket.blob(path)
    blob.content_type = "text/html"
    blob.upload_from_string(contents.encode(), content_type=blob.content_type)


def upload(html: str):
    hash = hashlib.md5(html.encode()).hexdigest()
    path = pathlib.Path(PATH) / (hash + ".html")
    url = f"http://storage.googleapis.com/{BUCKET}/{path}"
    if sys.stdin.isatty():
        ch = input(f"upload file to: {url} (Y/n)")
    else:
        ch = "y"
    if ch != "n":
        upload_html(str(path), html)
