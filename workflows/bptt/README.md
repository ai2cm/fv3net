Backpropagation Through Time
============================

This directory contains code and make targets to train a model using backpropagation through time.

This is done through the specified Makefile targets. Prognostic run targets must happen within the prognostic run image, which can be entered using `make dev_prognostic_run`. Other run targets must happen within the fv3net environment, which can be installed on your local machine or entered via the fv3net image.

In this training code, please be aware several model details have been hard-coded, such as having air_temperature and specific_humidity as prognostic variables, stored in that order within ML arrays. Input variables and training dataset details are stored as globals in `preprocessing.py`.
