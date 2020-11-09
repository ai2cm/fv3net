# Nudge-to-obs run workflow

This directory contains an example workflow for doing an FV3GFS run
that is nudged towards GFS analysis.

Some unique aspects of the given example:
- initialized 1 January 2016 from GFS analysis
- runs on 24 cores
- nudging tendencies and python outputs are every 5 hours
- 2D dycore and physics diagnostics output every 3 hours
- run is configured to output nudging tendencies as well as physics component tendencies

Given the default nudge-to-obs chunking and the output frequency specified in this example,
you should use segment lengths that are a multiple of five days.

Note that GFS analysis data is only available from 1 January 2015 to 1 January 2017, so
any portion of a model run that falls outside this period will not having nudging active.