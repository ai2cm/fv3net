export tag=test-run-$(openssl rand --hex 6)

export WANDB_RUN_GROUP=$tag

echo "Tag is $tag" > /dev/stderr
docker-compose run --rm -e WANDB_RUN_GROUP=$tag fv3 python3 scripts/prognostic_run.py --tag $tag
python3 scripts/piggy-back.py $tag
