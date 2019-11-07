#! /usr/bin/env sh

jobPrefix=rerun-fv3

ITERATION="$(date +%s)"
printf "iteration: %s\n" "$ITERATION" >&2

export ITERATION

while read time; do
  export timeForJobName=$(echo $time | sed 's/\.//g')
  export JOBNAME=$jobPrefix-$timeForJobName-$(uuid)
  export TIMESTEP="$time" && envsubst < job.yaml  | tee job.$time.yaml | kubectl apply -f -
  sleep 1
done < /dev/stdin

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
