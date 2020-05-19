from numpy.distutils.core import Extension, setup

mappm_extension = Extension(name="mappm", sources=["mappm.f90"])

setup(
    name="mappm",
    version="0.1.0",
    description="Python wrapper of mappm routine from FV3GFS model",
    author="Vulcan Technologies, LLC",
    author_email="oliwm@vulcan.com",
    ext_modules=[mappm_extension],
)
