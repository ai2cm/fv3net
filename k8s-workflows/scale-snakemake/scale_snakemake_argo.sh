#! /usr/bin/env sh

timesteps=$(bash list_timesteps_to_run.sh)

ITERATION="$(date +%s)"
printf "iteration: %s\n" "$ITERATION" >&2

export ITERATION

rm -f argo_jobs.yml
while read time; do
  export TIMESTEP="$time" && envsubst < argo.yml
  echo "---"
done < /dev/stdin
