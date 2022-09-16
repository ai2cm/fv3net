#!/bin/sh
set -e

# Assumes that environment variables are set already instead of taking arguments.
. $FV3NET_SCRIPTS/environment_variables.sh
exec "$@"
