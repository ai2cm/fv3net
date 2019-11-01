
# Install ARGO 

https://github.com/argoproj/argo/blob/master/demo.md

# Adding GCS credentials to K8s

https://kubernetes.io/docs/concepts/configuration/secret/

Download key for service acccount:

    gcloud iam service-accounts keys create --iam-account noah-vm-sa@vcm-ml.iam.gserviceaccount.com key.json

Add as a secret to the k8s cluster:

    kubectl create secret generic my-secret --from-file=./key.json

Then use it in a yaml file. Here is an example that can be incorporated into an argo workflow:

```yaml
# Need to install key
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: test-upload-
spec:
  entrypoint: process-restart
  volumes:
  - name: my-secret-vol
    secret:
      secretName: my-secret

  templates:
  - name: process-restart
    inputs:
    script:
      image: google/cloud-sdk
      env:
      - name: GOOGLE_APPLICATION_CREDENTIALS
        value: /secret/key.json
      command: [bash]
      source: |
        gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
        touch hello
        gsutil cp hello gs://vcm-ml-data/testing-touching.deleteme
      volumeMounts:
      - name: my-secret-vol
        mountPath: "/secret/"
```

TODO: I don't think the `gcloud auth` command should be necessary, but only setting the environmental variable GOOGLE_APPLICATION_CREDENTIALS does not seem to work.

In principle, this won't be necessary if we set up the service account for the cluster in the right way, potentially using terraform.

# Build and push the fv3net image to GCR

In the root of fv3net run 

    make push_image

# Run the argo workflow

Generate the yaml file describing all the jobs
      
    cd k8s-workflows/scale-snakemake/
    bash scale_snakemake_argo.sh

This generates a file `argo_jobs.yml`, which can be submitted like this:

    argo submit argo_jobs.yml 

