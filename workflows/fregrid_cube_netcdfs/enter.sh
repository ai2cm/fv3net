#!/bin/bash
# Utility script for entering a docker image with proper google credentials
# Used for developement purposes only

credentials=/etc/key.json
bindMounts="-v $credentials:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json  -v /home/oliwm:/home/oliwm -w $(pwd) -ti "
docker run $bindMounts --entrypoint /bin/bash us.gcr.io/vcm-ml/regrid_c3072_diag
