from setuptools import setup, find_packages


setup(
    name="derived",
    version="0.1.0",
    python_requires=">=3.7.0",
    author="Anna Kwa",
    author_email="annak@vulcan.com",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=["fsspec>=0.7.4", "xarray>=0.15.1", "vcm", "fv3gfs.util"],
    dependency_links=["../vcm", "../fv3gfs-util"],
)
