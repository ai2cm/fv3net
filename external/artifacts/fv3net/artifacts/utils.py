import fsspec


async def _list(fs: fsspec.AbstractFileSystem, path):
    try:
        return await fs._ls(path)
    except AttributeError:
        return fs.ls(path)


async def _close_session(fs):
    try:
        await fs.session.close()
    except AttributeError:
        pass


async def _cat_file(fs: fsspec.AbstractFileSystem, path, **kwargs):
    try:
        return await fs._cat_file(path, **kwargs)
    except AttributeError:
        return fs.cat_file(path, **kwargs)
