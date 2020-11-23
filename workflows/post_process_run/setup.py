from setuptools import setup, find_packages


with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name="fv3post",
    version="0.1.0",
    python_requires=">=3.7.0",
    author="Oliver Watt-Meyer",
    author_email="oliwm@vulcan.com",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=requirements,
    scripts=[
        "fv3post/scripts/fregrid_cubed_to_latlon.sh",
        "fv3post/scripts/fregrid_cubed_to_latlon_single_netcdf_input.sh",
    ],
    test_suite="tests",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "post_process_run=fv3post.post_process:post_process",
            "append_run=fv3post.append:append_segment",
        ]
    },
)
