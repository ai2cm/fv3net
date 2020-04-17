# VCM Experiment Configurations

## Purposes

This repository contains *declarative* configurations of our workflow, which should be cleanly separated from the source code in [fv3net]. Decoupling our source from the configuration will allow us to run experiments against different versions of the fv3net source. As the workflows in that repo become more decoupled and plug-n-play, this will allow us to change swap components out easily without being bound to the current master version of [fv3net].

## Dependencies

We should be able to manage our workflow configurations with only `kubectl` (v1.16.3). This tool is robust and stable and required for all kubrenetes work anyway. This tool set is intentionally limiting, but not as limiting as it would appear. `kubectl` provides a powerful templating tool called [kustomize] that currently suites our needs well. Any more detailed templating or configuration generation should probably occur inside of the k8s jobs we deploy. If it isn't human readable and editable, then it should be occuring inside of any k8s pods deployed by this repository.

## Structure

``` 
.
├── CODEOWNERS
├── README.md
├── base
│   ├── end_to_end.yaml
│   ├── integration-test-service-account.yml
│   ├── job.yaml
│   ├── kustomization.yaml
│   ├── one_step_times.json
│   └── training_times.json
├── base-full-sample
│   ├── kustomization.yaml
│   ├── one_step_times.json
│   └── training_times.json
└── integration-test
    ├── coarsen_c384_diagnostics_integration.yml
    ├── create_training_data_variable_names.yml
    ├── diag_table_prognostic
    ├── kustomization.yaml
    ├── one_step_jobs_integration.yml
    ├── prognostic_run_integration.yml
    ├── test_sklearn_variable_names.yml
    └── train_sklearn_model.yml
```

### Kustomization 

[kustomization] is a powerful templating system that is packaged with kubectl. It works by specifying a bases set of resources in `base/kustomization.yaml`, and then allow other workflows to inherit and modify this base configuration in a variety of ways (e.g. add configurations to a configmap, or add a suffix to the k8s job). Configurations to individual workflow steps are mostly controlled by the `.yml` files referred to within the `base/end_to_end.yaml` file. The following resources must be configured by editing the `kustomization.yaml` file in the root of the template directory. The settings in this file are overlayed on top of the configurations in `base/kustomization.yaml`. So go to that file to change settings shared by all the experiments (e.g. the prognostic run image tag and fv3net image tag).


## Workflow

1. Copy the "integration-test" folder
1. edit the workflow step configurations
1. change entries in the `<new experiments> kustomize.yaml` file:
    1. "nameSuffix" to change the name of the job
1. change global options like in the `base` directory.
1. Plan (optional): view the constructed k8s resources using `kubectl apply -k <dirname> --dry-run -o yaml`. `kustomize build <dirname>` will give cleaner output, but requires some [additional installation](https://github.com/kubernetes-sigs/kustomize/blob/master/docs/INSTALL.md).
1. Deploy the job with using `kustomize`: `kubectl apply -k <dirnmae>`


## Troubleshooting

This workflow *might* work if your versions deviate from the ones listed above. If you run into issues, run

    make versions

to show how your versions deviate from these.

### Trouble creating the job

If you get an error:

```
configmap/end-to-end-integration-test-g9m8dkg5mb configured
The Job "end-to-end-integration-test" is invalid: spec.template: Invalid value: core.PodTemplateSpec{ObjectMeta:v1.ObjectMeta{Name:"integration-test-pod", GenerateName:"", Namespace:"", SelfLink:"", UID:"", ResourceVersion:"", Generation:0, CreationTimestamp:v1.Time{Time:time.Time{wall:0x0, ext:0, loc:(*time.Location)(nil)}}, DeletionTimestamp:(*v1.Time)(nil), DeletionGracePeriodSeconds:(*int64)(nil), Labels:map[string]string{"controller-uid":"0a6491f2-8029-11ea-874b-42010a3c000c", "job-name":"end-to-end-integration-test"}, Annotations:map[string]string(nil), OwnerReferences:[]v1.OwnerReference(nil), Initializers:(*v1.Initializers)(nil), Finalizers:[]string(nil), ClusterName:"", ManagedFields:[]v1.ManagedFieldsEntry(nil)}, Spec:core.PodSpec{Volumes:[]core.Volume{core.Volume{Name:"gcp-key-secret", VolumeSource:core.VolumeSource{HostPath:(*core.HostPathVolumeSource)(nil), EmptyDir:(*core.EmptyDirVolumeSource)(nil), GCEPersistentDisk:(*core.GCEPersistentDiskVolumeSource)(nil), AWSElasticBlockStore:(*core.AWSElasticBlockStoreVolumeSource)(nil), GitRepo:(*core.GitRepoVolumeSource)(nil), Secret:(*core.SecretVolumeSource)(0xc005c74dc0), NFS:(*core.NFSVolumeSource)(nil), ISCSI:(*core.ISCSIVolumeSource)(nil), Glusterfs:(*core.GlusterfsVolumeSource)(nil), PersistentVolumeClaim:(*core.PersistentVolumeClaimVolumeSource)(nil), RBD:(*core.RBDVolumeSource)(nil), Quobyte:(*core.QuobyteVolumeSource)(nil), FlexVolume:(*core.FlexVolumeSource)(nil), Cinder:(*core.CinderVolumeSource)(nil), CephFS:(*core.CephFSVolumeSource)(nil), Flocker:(*core.FlockerVolumeSource)(nil), DownwardAPI:(*core.DownwardAPIVolumeSource)(nil), FC:(*core.FCVolumeSource)(nil), AzureFile:(*core.AzureFileVolumeSource)(nil), ConfigMap:(*core.ConfigMapVolumeSource)(nil), VsphereVolume:(*core.VsphereVirtualDiskVolumeSource)(nil), AzureDisk:(*core.AzureDiskVolumeSource)(nil), PhotonPersistentDisk:(*core.PhotonPersistentDiskVolumeSource)(nil), Projected:(*core.ProjectedVolumeSource)(nil), PortworxVolume:(*core.PortworxVolumeSource)(nil), ScaleIO:(*core.ScaleIOVolumeSource)(nil), StorageOS:(*core.StorageOSVolumeSource)(nil), CSI:(*core.CSIVolumeSource)(nil)}}, core.Volume{Name:"end-to-end-config", VolumeSource:core.VolumeSource{HostPath:(*core.HostPathVolumeSource)(nil), EmptyDir:(*core.EmptyDirVolumeSource)(nil), GCEPersistentDisk:(*core.GCEPersistentDiskVolumeSource)(nil), AWSElasticBlockStore:(*core.AWSElasticBlockStoreVolumeSource)(nil), GitRepo:(*core.GitRepoVolumeSource)(nil), Secret:(*core.SecretVolumeSource)(nil), NFS:(*core.NFSVolumeSource)(nil), ISCSI:(*core.ISCSIVolumeSource)(nil), Glusterfs:(*core.GlusterfsVolumeSource)(nil), PersistentVolumeClaim:(*core.PersistentVolumeClaimVolumeSource)(nil), RBD:(*core.RBDVolumeSource)(nil), Quobyte:(*core.QuobyteVolumeSource)(nil), FlexVolume:(*core.FlexVolumeSource)(nil), Cinder:(*core.CinderVolumeSource)(nil), CephFS:(*core.CephFSVolumeSource)(nil), Flocker:(*core.FlockerVolumeSource)(nil), DownwardAPI:(*core.DownwardAPIVolumeSource)(nil), FC:(*core.FCVolumeSource)(nil), AzureFile:(*core.AzureFileVolumeSource)(nil), ConfigMap:(*core.ConfigMapVolumeSource)(0xc005c74f40), VsphereVolume:(*core.VsphereVirtualDiskVolumeSource)(nil), AzureDisk:(*core.AzureDiskVolumeSource)(nil), PhotonPersistentDisk:(*core.PhotonPersistentDiskVolumeSource)(nil), Projected:(*core.ProjectedVolumeSource)(nil), PortworxVolume:(*core.PortworxVolumeSource)(nil), ScaleIO:(*core.ScaleIOVolumeSource)(nil), StorageOS:(*core.StorageOSVolumeSource)(nil), CSI:(*core.CSIVolumeSource)(nil)}}}, InitContainers:[]core.Container(nil), Containers:[]core.Container{core.Container{Name:"integration-test", Image:"us.gcr.io/vcm-ml/fv3net:latest", Command:[]string{"/bin/bash", "-c"}, Args:[]string{"echo \"running the following end to end configuration:\"\necho \"-------------------------------------------------------------------------------\"\nenvsubst $(CONFIG)/end_to_end.yml | tee end-to-end.yml\necho \"-------------------------------------------------------------------------------\"\nworkflows/end_to_end/submit_workflow.sh end-to-end.yml\n"}, WorkingDir:"", Ports:[]core.ContainerPort(nil), EnvFrom:[]core.EnvFromSource(nil), Env:[]core.EnvVar{core.EnvVar{Name:"GOOGLE_APPLICATION_CREDENTIALS", Value:"/secret/gcp-credentials/key.json", ValueFrom:(*core.EnvVarSource)(nil)}, core.EnvVar{Name:"CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", Value:"/secret/gcp-credentials/key.json", ValueFrom:(*core.EnvVarSource)(nil)}, core.EnvVar{Name:"PROGNOSTIC_RUN_IMAGE", Value:"", ValueFrom:(*core.EnvVarSource)(0xc00e566fa0)}, core.EnvVar{Name:"CONFIG", Value:"/etc/config", ValueFrom:(*core.EnvVarSource)(nil)}}, Resources:core.ResourceRequirements{Limits:core.ResourceList(nil), Requests:core.ResourceList{"cpu":resource.Quantity{i:resource.int64Amount{value:2, scale:0}, d:resource.infDecAmount{Dec:(*inf.Dec)(nil)}, s:"2", Format:"DecimalSI"}, "memory":resource.Quantity{i:resource.int64Amount{value:2, scale:9}, d:resource.infDecAmount{Dec:(*inf.Dec)(nil)}, s:"2G", Format:"DecimalSI"}}}, VolumeMounts:[]core.VolumeMount{core.VolumeMount{Name:"gcp-key-secret", ReadOnly:true, MountPath:"/secret/gcp-credentials", SubPath:"", MountPropagation:(*core.MountPropagationMode)(nil), SubPathExpr:""}, core.VolumeMount{Name:"end-to-end-config", ReadOnly:false, MountPath:"/etc/config/", SubPath:"", MountPropagation:(*core.MountPropagationMode)(nil), SubPathExpr:""}}, VolumeDevices:[]core.VolumeDevice(nil), LivenessProbe:(*core.Probe)(nil), ReadinessProbe:(*core.Probe)(nil), Lifecycle:(*core.Lifecycle)(nil), TerminationMessagePath:"/dev/termination-log", TerminationMessagePolicy:"File", ImagePullPolicy:"Always", SecurityContext:(*core.SecurityContext)(nil), Stdin:false, StdinOnce:false, TTY:false}}, RestartPolicy:"Never", TerminationGracePeriodSeconds:(*int64)(0xc007582de0), ActiveDeadlineSeconds:(*int64)(nil), DNSPolicy:"ClusterFirst", NodeSelector:map[string]string(nil), ServiceAccountName:"integration-tests", AutomountServiceAccountToken:(*bool)(nil), NodeName:"", SecurityContext:(*core.PodSecurityContext)(0xc015d50a10), ImagePullSecrets:[]core.LocalObjectReference(nil), Hostname:"", Subdomain:"", Affinity:(*core.Affinity)(nil), SchedulerName:"default-scheduler", Tolerations:[]core.Toleration{core.Toleration{Key:"dedicated", Operator:"", Value:"climate-sim-pool", Effect:"NoSchedule", TolerationSeconds:(*int64)(nil)}}, HostAliases:[]core.HostAlias(nil), PriorityClassName:"", Priority:(*int32)(nil), DNSConfig:(*core.PodDNSConfig)(nil), ReadinessGates:[]core.PodReadinessGate(nil), RuntimeClassName:(*string)(nil), EnableServiceLinks:(*bool)(nil)}}: field is immutable
```

This mess means that the job-name already exists in the cluster. You could delete it with `kubectl delete job <jobname>`, but it is probably preferable to keep the job in the cluster for posterity. For production use, it is better to change the `nameSuffix` in the `kustomization.yaml` folder, which will generate a unique name.


[fv3net]: https://github.com/VulcanClimateModeling/fv3net
[kustomize]: https://kustomize.io/ 
