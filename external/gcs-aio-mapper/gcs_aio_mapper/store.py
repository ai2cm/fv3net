from typing import Hashable, Mapping
import aiohttp
import asyncio
from gcloud.aio.storage import Storage
from google.cloud.storage import Client
from collections import MutableMapping
import os
import logging

logger = logging.getLogger(__package__)


async def _upload_obj(client, bucket, prefix, key, val):
    logging.debug(f"uploading {key} to {bucket}/{prefix}")
    location = os.path.join(prefix, key)
    return await client.upload(bucket, location, val)


async def _upload(cache: Mapping[str, bytes], bucket, prefix):
    items = list(cache.items())
    futures = []
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)

        while len(items) > 0:
            key, val = items.pop()
            futures.append(_upload_obj(client, bucket, prefix, key, val))

            if len(futures) > 10 or len(items) == 0:
                await asyncio.gather(*futures)


async def delete_items(bucket, items):
    async with aiohttp.ClientSession() as session:
        logging.debug(f"deleting {items} in {bucket}")
        client = Storage(session=session)
        return await asyncio.gather(*[client.delete(bucket, item) for item in items])


async def _get(bucket, prefix, key):
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)
        return await client.download(bucket, os.path.join(prefix, key))


async def _delete_item(bucket, item):
    async with aiohttp.ClientSession() as session:
        logging.debug(f"deleting {item} in {bucket}")
        client = Storage(session=session)
        return await client.delete(bucket, item)


class GCSMapperAio(MutableMapping):
    def __init__(self, url, cache_size=20, cache=None, project=None):
        super().__init__()
        self._cache = {} if cache is None else cache
        self._url = url
        self.project = project
        self.cache_size = cache_size

    def __getitem__(self, key: Hashable):
        key = self._normalize_key(key)
        if key in self._cache:
            return self._cache[key]
        else:
            try:
                return self._getitem_remote(key)
            except:
                raise KeyError

    def _getitem_remote(self, key: Hashable):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_get(self.bucket, self.prefix, key))

    @property
    def bucket(self):
        return self._url.lstrip('gs://').split('/')[0]

    @property
    def prefix(self):
        return '/'.join(self._url.lstrip('gs://').split('/')[1:])

    def _upload_cache_to_remote(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_upload(self._cache, self.bucket, self.prefix))
        self._cache = {}

    def _list_remote_keys(self):
        client = Client(project=self.project)
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            yield blob.name[len(self.prefix)+1:]

    def keys(self):
        for key in list(self._cache):
            yield key
        yield from self._list_remote_keys()

    def __setitem__(self, key: Hashable, val: bytes):
        key = self._normalize_key(key)
        self._cache[key] = val
        if len(self._cache) > self.cache_size:
            self.flush()

    def __delitem__(self, key):
        key = self._normalize_key(key)
        if key in self._cache.keys():
            del self._cache[key]
        else:
            loop = asyncio.get_event_loop()
            out = os.path.join(self.prefix, key)
            return loop.run_until_complete(_delete_item(self.bucket, out))

    def __iter__(self):
        return self.keys()

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.flush()

    def __len__(self):
        return len(self.keys())

    def flush(self):
        self._upload_cache_to_remote()

    def _key_path(self, key: str):
        return os.path.join(self.prefix, key)

    def _normalize_key(self, key):
        return str(key)

    def rmdir(self, path=None):
        remote_keys = []
        for key in self.keys():
            if path is None or key.startswith(path):
                try:
                    del self._cache[key]
                except KeyError:
                    remote_keys.append(self._key_path(key))
        op = delete_items(self.bucket, remote_keys)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(op)
