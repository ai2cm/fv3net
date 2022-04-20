Based on this tutorial: https://argoproj.github.io/argo-events/tutorials/01-introduction/


To setup all the stuff

    kustomize build . | kubectl apply -f -
    # might error on the CRDs...run it again in 5 seconds
    kustomize build . | kubectl apply -f -

Wait a minute or two. The submit a workflow

    argo submit workflow.yaml

Listing the workflows should show two workflows:

    $ argo list
    NAME                      STATUS      AGE   DURATION   PRIORITY
    resource-workflow-5h4fz   Succeeded   49s   45s        0
    my-workflow54mc5          Succeeded   49s   44s        0

The "resource-workflow-" workflow was launched when the first workflow was
detected. See [./sensor-resource.yaml]. This is the data it was passed:
```
$ argo logs resource-workflow-5h4fz
 _________________________________________
/ {"apiVersion":"argoproj.io/v1alpha1","k \
| ind":"Workflow","metadata":{"creationTi |
| mestamp":"2022-04-20T22:46:30Z","genera |
| teName":"my-workflow","generation":1,"l |
| abels":{"app":"my-workflow"},"managedFi |
| elds":[{"apiVersion":"argoproj.io/v1alp |
| ha1","fieldsType":"FieldsV1","fieldsV1" |
| :{"f:metadata":{"f:generateName":{},"f: |
| labels":{".":{},"f:app":{}}},"f:spec":{ |
| },"f:status":{}},"manager":"argo","oper |
| ation":"Update","time":"2022-04-20T22:4 |
| 6:30Z"}],"name":"my-workflow54mc5","nam |
| espace":"default","resourceVersion":"10 |
| 25","uid":"4cbf972c-071e-4f0f-a307-290a |
| 2ea60505"},"spec":{"arguments":{},"entr |
| ypoint":"whalesay","templates":[{"conta |
| iner":{"args":["hello                   |
| world"],"command":["cowsay"],"image":"d |
| ocker/whalesay:latest","name":"","resou |
| rces":{}},"inputs":{},"metadata":{},"na |
| me":"whalesay","outputs":{}}]},"status" |
\ :{"finishedAt":null,"startedAt":null}}  /
 -----------------------------------------
    \
     \
      \
                    ##        .
              ## ## ##       ==
           ## ## ## ##      ===
       /""""""""""""""""___/ ===
  ~~~ {~~ ~~~~ ~~~ ~~~~ ~~ ~ /  ===- ~~~
       \______ o          __/
        \    \        __/
          \____\______/
```
