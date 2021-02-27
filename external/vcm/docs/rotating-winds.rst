Converting from x/y winds to lat/lon
====================================

The prognostic horizontal winds in the FV3 dynamical core are oriented in a
local x/y coordinate system on the cell edges of a cubed-sphere grid.  The
functions defined here can be used to convert the winds defined in that
coordinate system to a be oriented in a latitude-longitude coordinate system.

.. automethod:: vcm.cubedsphere.rotate.center_and_rotate_xy_winds
.. automethod:: vcm.cubedsphere.rotate.rotate_xy_winds