import sys
from google.cloud import storage

def parse_gs_string(url):
    stripped = url.lstrip('gs://')
    for k, ch in enumerate(stripped):
        if ch == '/':
            break
    return stripped[:k], stripped[k+1:]

bucket_name, blob_name = parse_gs_string(sys.argv[2])

client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = storage.Blob(blob_name, bucket)
blob.upload_from_filename(sys.argv[1], content_type='text/html')
