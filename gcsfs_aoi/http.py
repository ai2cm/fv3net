import asyncio
import aiohttp
import os


async def get(session: aiohttp.ClientSession, url):
    async with session.get(url) as response:
        print(f"getting {url}")
        return await response.text()


# async def get(session, url):
#     print(f"starting {url}")
#     await asyncio.sleep(url)
#     print(f"done {url}")


async def getmany(urls):
    async with aiohttp.ClientSession() as session:
        futures = [get(session, url) for url in urls]
        print(await asyncio.gather(*futures))


urls = ["http://www.google.com", "http://www.bing.com"]
loop = asyncio.get_event_loop()
loop.run_until_complete(getmany(urls))
