.. _quickstarts:

Quickstarts
===========

Here are some quick setup items for running and developing with ``fv3net``.

Installation
------------

To install all the requirements and fv3net packages locally, run this from the main directory::

    make update_submodules
    make create_environment

This creates an Anaconda environment ``fv3net`` with all the dependicies for running the Vulcan FV3GFS ML workflows.   All submodules are installed in development mode (e.g., ``pip install -e``) so any modifications will be loaded in the conda environment.  After the build completes successfully, activate the environment with::

    conda activate fv3net

.. _cloud_auth:

Cloud Authentication
--------------------

The `fv3net` project currently utilizes Google Cloud to deploy workflow items to services such as Kubernetes and Dataflow.  Authentication requires an installation of `Google Cloud SDK <https://cloud.google.com/sdk/docs/install>`_.

Authentication obtained via ``gcloud auth login`` does not work well with secrets management and is not used by many APIs. Service account key-based authentication works much better, because the service account key is a single file that can be deployed in a variety of contexts (K8s cluter, VM, etc) (`Google docs on service accounts <https://cloud.google.com/iam/docs/service-accounts>`_). Many Python APIs can authenticate with google using the ``GOOGLE_APPLICATION_CREDENTIALS`` environmental variable `(See Google authentication details) <https://cloud.google.com/sdk/docs/authorizing>`_.

* If gcloud is a fresh install, initialize and grab a keyfile::
      
    > gcloud init
    > gcloud auth login
    > mkdir -p ~/.keys
    > gcloud iam service-accounts keys create ~/.keys/key.json \
          --iam-account <service account>

* Else activate your service account key::

    > gcloud auth activate-service-account <account> --key-file=<key-file>
    > export GOOGLE_APPLICATION_CREDENTIALS=~/.keys/key.json

.. note::

    It is recommended to add ``GOOGLE_APPLICATION_CREDENTIALS`` to your .bashrc since many libraries and tools require it.

Connecting to a kubernetes cluster
----------------------------------

  * Pre-existing basic cluster::

      > gcloud container clusters get-credentials <cluster-name>

  * From a VM to our firewalled cluster (Vulcan Specific)

    * Clone the `long-lived-infrastructure repo <https://github.com/VulcanClimateModeling/long-lived-infrastructure>`_
    * Use terraform to connect to our cluster `(details) <https://github.com/VulcanClimateModeling/long-lived-infrastructure#vm-access-setup>`_::
        
        > make tf_init
        > make tf_dev_workspace_create
        > make kubeconfig_init
    
  * We also have an detailed explanation of creating and connecting to a local k8s cluster in :ref:`local_k8s`. 

After authenticated you will be able to set up and utilize infrastructure with proper permissions. If you are having trouble with authentication being recognized check out the :ref:`faqs` page.

Install Argo
------------

Argo is a workflow engine for easier orchestration of jobs on Kubernetes. Use `these instructions <https://github.com/argoproj/argo-workflows/blob/master/docs/quick-start.md>`_ to install Argo on your workstation.

Add Argo templates
^^^^^^^^^^^^^^^^^^

If this is a new/personal cluster, then the ``fv3net`` argo templates can be installed using::

    kubectl apply -k workflows/argo

For more information, please see the :doc:`Argo README <readme_links/argo_readme>`.


Test drive
----------

For a simple test drive of authentication and cloud-native infrastructure we'll run a test of the prognostic run used by our end-to-end testing suite.  

This example submits a "baseline" run of our fv3gfs model run to the Kubernetes server using the Argo workflow template. First, it creates a configuration file ``test_fv3config.yaml`` needed as a parameter to the template, and then it submits the job to Kubernetes.

.. code-block:: bash

  cat <<"EOF" > test_fv3config.yaml
  base_version: v0.5
  namelist:
    coupler_nml:
      days: 0
      hours: 3
      minutes: 0
      seconds: 0
    diag_manager_nml:
      flush_nc_files: true
    fv_core_nml:
      do_sat_adj: false
    gfdl_cloud_microphysics_nml:
      fast_sat_adj: false
  EOF

  argo submit \
      --from workflowtemplate/prognostic-run \
      -p output=gs://vcm-ml-scratch/test-prognostic-run-example \
      -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
      -p initial-condition="20160805.000000" \
      -p config="$(cat ./test_fv3config.yaml)" \
      -p segment-count=1

After the job submits, there will be a read out of the job::

    Name:                prognostic-run-xk4nj
    Namespace:           default
    ServiceAccount:      default
    Status:              Pending
    Created:             Tue Feb 23 00:12:20 +0000 (now)
    Parameters:          
      output:            gs://vcm-ml-scratch/test-prognostic-run-example
      reference-restarts: gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
      initial-condition: 20160805.000000
      config:            base_version: v0.5
    namelist:
      coupler_nml:
        days: 0
        hours: 3
        minutes: 0
        seconds: 0
      diag_manager_nml:
        flush_nc_files: true
      fv_core_nml:
        do_sat_adj: false
      gfdl_cloud_microphysics_nml:
        fast_sat_adj: false
      segment-count:     1

And you can check on the job status using either ``argo get <job_name>`` or ``argo logs <job_name>``.

.. note::

    The prognostic run usage and configurability is a deep topic on its own.  Take a look at the `Prognostic run documentation <https://vulcanclimatemodeling.com/docs/prognostic_c48_run/>`_ to delve further into its abilities.  For other simple examples of submitting argo workflows, check out the `examples folder <https://github.com/VulcanClimateModeling/vcm-workflow-control/tree/master/examples>`_ in `vcm-workflow-control <https://github.com/VulcanClimateModeling/vcm-workflow-control>`_.

Cloud Workflows
---------------

The main data processing pipelines for this project currently utilize Kubernetes with Docker images and Dataflow on Google Cloud. Check out :ref:`workflows` to see how to run and compose them! The Makefiles typically specify what's being run with extended descriptions of the workflow in the ``README.md`` files.
