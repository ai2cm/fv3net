from setuptools import setup, find_packages

dependencies = [
    "apache-beam==2.16.0",
    "cloudpickle==1.2.2",
    "dask==2.6.0",
    "fsspec==0.5.2",
    "google-cloud-storage==1.20.0",
    "netCDF4==1.4.2",
    "partd",
]
setup(
    name="extract_tars",
    packages=find_packages(),
    install_requires=dependencies,
    version="0.1.0",
    description="Dataflow operation for batch untarring timesteps",
    author="Vulcan Inc.",
    license="MIT",
)
