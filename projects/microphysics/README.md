# Microphysics Emulation


## Prognostic Evaluation

Setup environment file to authenticate against external services
```
cat << EOF> .env
GOOGLE_CLOUD_PROJECT=vcm-ml
WANDB_API_KEY=<wandb api key>
EOF

# login to google cloud
gcloud auth login application-default login
```

Pull docker image

    TAG=latest # replace with the desired git sha
    docker pull us.gcr.io/vcm-ml/prognostic_run:$TAG
    # tag with latest to use with docker-compose
    docker tag us.gcr.io/vcm-ml/prognostic_run:$TAG us.gcr.io/vcm-ml/prognostic_run:latest

Enter the docker image

    docker-compose run --rm fv3

Run the prognostic run

    python3 scripts/prognostic_run.py --duration 1h

Pass `--help` to this script for more information


