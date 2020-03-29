from typing import Hashable, Mapping
import aiohttp
import asyncio
from gcloud.aio.storage import Storage
from collections import MutableMapping
import gcsfs
import os


async def _upload_obj(client, bucket, prefix, key, val):
    location = os.path.join(prefix, key)
    return await client.upload(bucket, location, val)


async def _upload(cache: Mapping[str, bytes], bucket, prefix):
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)
        all_ops = await asyncio.gather(
            *[
                _upload_obj(client, bucket, prefix, key, val)
                for key, val in cache.items()
            ]
        )
        # TODO check that ops were succesful?
        return all_ops


async def delete_items(bucket, items):
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)
        return await asyncio.gather(*[client.delete(bucket, item) for item in items])


async def _get(bucket, prefix, key):
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)
        return await client.download(bucket, os.path.join(prefix, key))


async def _delete_item(bucket, item):
    async with aiohttp.ClientSession() as session:
        client = Storage(session=session)
        return await client.delete(bucket, item)


class GCSFSMapperAoi(MutableMapping):
    def __init__(self, url, cache_size=20):
        super().__init__()
        self._cache = {}
        self._url = url
        self.cache_size = cache_size

    def __getitem__(self, key: Hashable):
        key = str(key)
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

    @property
    def _sync_mapper(self):
        return gcsfs.GCSMap(os.path.join(self.bucket, self. prefix))

    def _upload_cache_to_remote(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_upload(self._cache, self.bucket, self.prefix))
        self._cache = {}

    def keys(self):
        for key in self._cache:
            yield key
        for key in self._sync_mapper:
            yield key

    def __setitem__(self, key: Hashable, val: bytes):
        key = str(key)
        self._cache[key] = val
        if len(self._cache) > self.cache_size:
            self.flush()

    def __delitem__(self, key):
        if key in self._cache:
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

    def rmdir(self, root):
        remote_keys = []
        for key in self.keys():
            if key.startswith("root"):
                try:
                    del self._cache[key]
                except KeyError:
                    remote_keys.append(self._key_path(key))
        op = delete_items(self.bucket, remote_keys)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(op)


