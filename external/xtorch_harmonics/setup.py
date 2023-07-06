from setuptools import setup, find_packages


setup(
    name="xtorch_harmonics",
    version="0.1.0",
    python_requires=">=3.8",
    author="Spencer Clark and Oliver Watt-Meyer",
    author_email="spencerc@allenai.org",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "dask>=2022.11.1",
        "numpy>=1.12.1",
        "torch>=1.12.1",
        "torch_harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@8826246cacf6c37b600cdd63fde210815ba238fd",  # noqa: E501
        "xarray>=0.19.0",
    ],
)
