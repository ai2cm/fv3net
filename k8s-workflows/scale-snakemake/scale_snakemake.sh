#! /usr/bin/env sh

ITERATION="$(date +%s)"
printf "iteration: %s\n" "$ITERATION" >&2

for i in $( seq 0 10 ); do
  export I="$i" && export ITERATION="$ITERATION" && envsubst '$I:$ITERATION' < job.yaml | kubectl apply -f -
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
