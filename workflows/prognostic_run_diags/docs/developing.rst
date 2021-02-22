.. _developing:

Developing this workflow
========================

This system is designed for extension without extensively modifying the
source code. To accomplish this, it divides report generation into
phases:

#. Diagnostic computation: "diagnostic" is a multidimensional quantity 
   that is reasonably limited in size (e.g. can be easily downloaded).
#. Metric computation: scalar "metrics" calculated from the output of the
   step above.
#. Movie creation: generate move stills and then create movie with ffmpeg.
#. Report generation: multiple sets of diagnostics/metrics are visualized
   in a static html report.

Adding a diagnostic
~~~~~~~~~~~~~~~~~~~

A new diagnostics can be added by writing a new function in
:py:mod:`fv3net.diagnostics.prognostic_run.compute` and decorating it
with ``@add_to_diags`` and other desired transforms.

Adding a new metric
~~~~~~~~~~~~~~~~~~~

Similarly, a new metric can be added using the ``@add_to_metrics`` decorator in
:py:mod:`fv3net.diagnostics.prognostic_run.metrics`.

Adding a new visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The steps above simply add a metric or diagnostic to the saved output, but do
not change the report.

To plot a new metric or diagnostic, add a function to
:py:mod:`fv3net.diagnostics.prognostic_run.views.static_report` and decorate
it with the ``@metrics_plot_manager.register`` if its metric or
``@..._plot_manager.register`` for the desired report section if it is a diagnostic.
This function needs to return an object that is valid input to the ``sections`` argument of
``report.create_html``.

Testing
~~~~~~~

An integration test of the various steps required to generate a prognostic run report 
can be launched by calling ``make test_prognostic_run_report`` from the root of the 
fv3net repository.