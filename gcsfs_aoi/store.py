from typing import Hashable, Mapping
import aiohttp
import asyncio
from gcloud.aio.storage import Storage
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


class GCSFSMapperAoi:
    def __init__(self, url, cache_size=10):
        super().__init__()
        self._cache = {}
        self._url = url
        self.cache_size = cache_size

    def __getitem__(self, key: Hashable):
        if key in self._cache:
            return self._cache[key]
        else:
            return self._getitem_remote(key)

    def _getitem_remote(self, key: Hashable):
        pass

    @property
    def bucket(self):
        return self._url.lstrip('gs://').split('/')[0]

    @property
    def prefix(self):
        return '/'.join(self._url.lstrip('gs://').split('/')[1:])

    def _upload_cache_to_remote(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_upload(self._cache, self.url))
        pass

    def __setitem__(self, key: Hashable, val: bytes):
        self._cache[key] = val
        if len(self._cache) > self.cache_size:
            self._upload_cache_to_remote()

    def __del__(self):
        self._upload_cache_to_remote()

    def flush(self):
        self._upload_cache_to_remote()
