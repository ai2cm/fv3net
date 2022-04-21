Based on this tutorial: https://argoproj.github.io/argo-events/tutorials/01-introduction/

If developing on minikube you may need to startup the cluster:

    ./setup_cluster.sh

To setup all the stuff

    # Start here if deploying elswhere
    kustomize build manifests | kubectl apply -f -
    # might error on the CRDs...run it again in 5 seconds
    kustomize build manifests | kubectl apply -f -

Wait a minute or two (make sure everything is "Runnning" in `kubectl -n
argo-events pod`). Then submit a workflow

    argo submit workflow.yaml

Now print the logs of the server that transfers pod info to big query.

    $ kubectl logs -l app=bq
    2022/04/21 21:00:21 request received
    2022/04/21 21:00:21 {"apiVersion":"argoproj.io/v1alpha1","kind":"Workflow","metadata":{"creationTimestamp":"2022-04-21T21:00:21Z","generateName":"my-workflow","generation":1,"labels":{"app":"my-workflow"},"managedFields":[{"apiVersion":"argoproj.io/v1alpha1","fieldsType":"FieldsV1","fieldsV1":{"f:metadata":{"f:generateName":{},"f:labels":{".":{},"f:app":{}}},"f:spec":{},"f:status":{}},"manager":"argo","operation":"Update","time":"2022-04-21T21:00:21Z"}],"name":"my-workflow4zkq2","namespace":"default","resourceVersion":"1561","uid":"c7fb5177-8c96-4b17-a750-4eefd52aac60"},"spec":{"arguments":{},"entrypoint":"whalesay","templates":[{"container":{"args":["hello world"],"command":["cowsay"],"image":"docker/whalesay:latest","name":"","resources":{}},"inputs":{},"metadata":{},"name":"whalesay","outputs":{}}]},"status":{"finishedAt":null,"startedAt":null}}
    2022/04/21 21:00:22 Uploaded c7fb5177-8c96-4b17-a750-4eefd52aac60 to big query