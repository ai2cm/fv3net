#!/bin/bash

rm /usr/local/bin/runfv3
ln -s /fv3net/workflows/prognostic_c48_run/runfv3 /usr/local/bin/runfv3
chmod +x /usr/local/bin/runfv3
pip install -e /fv3net/external/fv3fit -c /fv3net/constraints.txt
bash
