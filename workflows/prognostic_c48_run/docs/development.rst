.. _development:

Developer's Guide
-----------------

The prognostic run is developed via docker. This environment is based off the
`prognostic_run` docker image, but has bind-mounts to the packages in "/external"
of this repository and this directory, which allows locally developing this workflow
and its dependencies.

It is usually fastest to use the latest docker image from Google Container
Repository. Pull the image::

    make pull_image_prognostic_run


If you run into problems, it would be best to rebuild the docker image from scratch::

    make build_image_prognostic_run

To build the image with the python radiation port available, run::

    PROGNOSTIC_RUN_RADIATION=y make build_image_prognostic_run

Enter a bash shell in the image::

    make enter_prognostic_run

Or::

    make enter_prognostic_run_radiation

Then run the tests::

    pytest