#!/bin/bash -e

set -e


function runRemote {

  set +e
  version=$(git rev-parse HEAD)
  if [[ $? -ne 0 ]]
  then
    version="$COMMIT_SHA"
  fi
  set -e

  image=us.gcr.io/vcm-ml/dataflow:"$version"
  python3 $* --experiments=use_runner_v2 --sdk_container_image=$image
}

function usage {
  echo "Submit a dataflow job"
  echo ""
  echo "Usage:"
  echo "  dataflow.sh submit (-m <module> | <absolute_path>) <args>..."
  echo "  dataflow.sh -h"
  echo ""
  echo "Commands:"
  echo ""
  echo "  submit     submit a remote dataflow job"
  echo ""
  echo "Options:"
  echo "  -h         Show the help"
}

function prepareWorkingDirectory {
  workdir=$(mktemp -d)
  buildPackages "$workdir/dists/"
  echo "$workdir"
}

if [[ $# -lt 1 ]]
then
  usage
  exit 2
fi


subcommand="$1"
shift

case $subcommand in
  submit)
    runRemote "$@"
    ;;
  -h)
    usage
    exit 2
    ;;
  *)
    >&2 echo "invalid subcommand: $subcommand"
    exit 2
    ;;
esac
