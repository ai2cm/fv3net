.. _faqs:

FAQs
====

Q: I can't submit an argo job on my VM! I'm getting an error like::

    (fv3net) mcgibbon@jeremy-vm:~/python/vcm-workflow-control/examples/nudge-to-fine-run$ ./run.sh
    2021/02/23 20:53:31 Failed to submit workflow: Get https://35.225.51.240/apis/argoproj.io/v1alpha1/namespaces/default/workflowtemplates/prognostic-run: error executing access token command "/snap/google-cloud-sdk/160/bin/gcloud config config-helper --format=json": err=fork/exec /snap/google-cloud-sdk/160/bin/gcloud: no such file or directory output= stderr=

A: Check out the `long lived infrastructure repo <https://github.com/VulcanClimateModeling/long-lived-infrastructure>`_ and run ``make kubeconfig_init``, then try again.

Authentication
--------------

See :ref:`cloud_auth` for details.

* My authentication to gcloud resources doesn't seem to be working
    Make sure that the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable
    exists and is properly linked to your service account keyfile.
* ``gsutil`` says I am an anonymous user and I can't access a public GCS bucket
    Ensure that there aren't conflicting Python and system installations of
    ``gsutil``. We recommend uninstalling the Python package if you find it.
    Otherwise, the fastest method to resolution if this is a one-time access
    is to flip the bucket switch for "Requester Pays" to be off temporarily.


Development
-----------


* I'm getting an error about a submodule directory under  ``external/`` being empty
    .. code-block:: bash

        > git submodule update --init --recursive”

