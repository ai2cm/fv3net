#!/bin/bash
# Utility script for entering a docker image with proper google credentials
# Used for developement purposes only

credentials=$HOME/keys/noahb-vm.json
bindMounts="-v $credentials:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json  -v /home/noahb:/home/noahb -w $(pwd) -ti "
docker run $bindMounts us.gcr.io/vcm-ml/fretools  bash
