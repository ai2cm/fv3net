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
    install_requires=["fsspec>=0.7.4", "pyyaml>=5.3.0", "kubernetes>=11", "fv3config"],
    dependency_links=["../fv3config"],
    entry_points={
        "console_scripts": [
            "prepare_config_nudging = fv3kube.prepare_config.nudging:main",
            "prepare_config_nudge_to_obs = fv3kube.prepare_config.nudge_to_obs:main",
            "prepare_config_prognostic_run = fv3kube.prepare_config.prognostic_run:main", # noqa
        ]
    },
)
