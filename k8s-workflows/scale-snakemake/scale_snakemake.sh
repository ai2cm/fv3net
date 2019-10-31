#! /usr/bin/env sh

#timesteps=$(bash list_timesteps_to_run.sh)
timesteps="20160801.003000 20160801.004500"

ITERATION="$(date +%s)"
printf "iteration: %s\n" "$ITERATION" >&2

export ITERATION

for time in $timesteps; do
  export timeForJobName=$(echo $time | sed 's/\.//g')
  export JOBNAME=snakemake-$timeForJobName
  export TIMESTEP="$time" && envsubst < job.yaml  | tee job.$time.yaml | kubectl apply -f -
  sleep 1
done

while true; do
  active_jobs="$(kubectl get jobs --selector "run=$ITERATION" --output=json | jq --raw-output '[.items[] | select (.status.active == 1)] | .[].metadata.name')" || exit 1
  if [ -z "$active_jobs" ]; then
    printf "all done!\n"
    break
  fi
  printf "active jobs at %s:\n%s\n" "$(date +%s)" "$active_jobs"
  sleep 10
done;

FAILED_JOBS="$(kubectl get jobs --selector "run=$ITERATION" --output=json | jq --raw-output '[.items[] | select (.status.failed == 1)] | .[].metadata.name')" || exit 1
printf "failed jobs from iteration %s:\n%s\n" "$ITERATION" "$FAILED_JOBS"

exit 0
