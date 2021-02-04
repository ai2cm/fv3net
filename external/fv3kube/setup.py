from setuptools import setup, find_packages


setup(
    name="fv3kube",
    version="0.1.0",
    python_requires=">=3.6.9",
    author="Oliver Watt-Meyer",
    author_email="oliwm@vulcan.com",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={"fv3kube": ["base_yamls/*/fv3config.yml"]},
    install_requires=["fsspec>=0.7.4", "pyyaml>=5.3.0", "kubernetes>=11", "fv3config"],
    dependency_links=["../fv3config"],
)
