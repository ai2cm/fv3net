from abc import ABC, abstractmethod
from typing import Iterable

from google.cloud.storage import Client  # type: ignore


class FileLister(ABC):
    @abstractmethod
    def list(self, prefix=None, file_extension=None) -> Iterable[str]:
        pass


class GCSLister(FileLister):
    def __init__(self, client: Client, bucket: str):
        self.client = client
        self.bucket = bucket

    def list(self, prefix=None, file_extension=None) -> Iterable[str]:
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        for blob in blobs:
            
            # filter specific extensions
            if file_extension is not None:
                blob_ext_name = blob.name.split('.')[-1]
                if file_extension.strip('.') != blob_ext_name:
                    continue
            
            # TODO: Should pass the blob not the GS filename....
            yield f"gs://{blob.bucket.name}/{blob.name}"