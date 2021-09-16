resources(name="impl", sources=["*.f90"])
python_library()
python_requirement_library(name="numpy", requirements=["numpy"])
python_distribution(
    name="mappm-dist",
    dependencies = [":impl", ":mappm", ":numpy"],
    provides=python_artifact(
        setup_script="setup.py",
        name="mappm",
        version="0.1.0",
    ),
    setup_py_commands = ["bdist_wheel"]
)