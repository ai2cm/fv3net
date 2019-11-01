#! /usr/bin/env sh

timesteps=$(bash list_timesteps_to_run.sh)
timesteps=20160803.021500

ITERATION="$(date +%s)"
printf "iteration: %s\n" "$ITERATION" >&2

export ITERATION

rm -f argo_jobs.yml
for time in $timesteps; do
  export TIMESTEP="$time" && envsubst < argo.yml >> argo_jobs.yml
  echo "---" >> argo_jobs.yml
done
