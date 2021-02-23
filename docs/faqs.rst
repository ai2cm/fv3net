.. _faqs:

FAQs
====


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

