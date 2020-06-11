from setuptools import setup, find_packages


setup(
    name="loaders",
    version="0.1.0",
    python_requires=">=3.7.0",
    author="Anna Kwa",
    author_email="annak@vulcan.com",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=[
        "fsspec>=0.7.4",
        "numpy>=1.18.4",
        "pyyaml>=5.3.0",
        "toolz>=0.10.0",
        "typing>=3.7.4",
        "xarray>=0.15.1",
        "zarr>=2.4.0",
        "vcm",
    ],
    dependency_links=["../vcm"],
)
