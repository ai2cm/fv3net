from setuptools import setup, find_packages


setup(
    name="fv3kube",
    version="0.1.0",
    python_requires=">=3.7.0",
    author="Oliver Watt-Meyer",
    author_email="oliwm@vulcan.com",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=[
        "fsspec>=0.7.4",
        "numpy>=1.18.4",
        "pyyaml>=5.3.0",
        "kubernetes>=11",
        "vcm",
        "fv3config",
    ],
    dependency_links=["../vcm", "../fv3config"],
)
