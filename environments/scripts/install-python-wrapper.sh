#!/bin/bash

cd ../../external/fv3gfs-fortran/FV3
LDSHARED="cc -shared" make wrapper_build
pip install --no-dependencies wrapper/dist/*whl --force-reinstall
