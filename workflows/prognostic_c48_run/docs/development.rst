.. _development:

Developer's Guide
-----------------

The prognostic run is developed via docker and docker-compose. It is usually fastest to use the
latest docker image from Google Container Repository. Pull the image::

    docker pull us.gcr.io/vcm-ml/prognostic_run:latest

.. note::

    If you run into problems, it would be best to rebuild the docker image from scratch::

        docker-compose build fv3


Enter a bash shell in the image::

    docker-compose run fv3net bash

.. note :: 

    This docker-compose will propagate key-based authentication to Google
    Cloud Platform into the docker image. It expects that environmental variable 
    ``GOOGLE_APPLICATION_CREDENTIALS`` points to a json key. See Google's
    `documentation <https://cloud.google.com/iam/docs/creating-managing-service-account-keys>`_
    on how to generate one.
    
Run the tests::

    pytest