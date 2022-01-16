Thermodynamics
==============

Atmospheric pressure
--------------------

   .. autofunction:: vcm.calc.thermo.vertically_dependent.pressure_at_midpoint
   .. autofunction:: vcm.pressure_at_midpoint_log
   .. autofunction:: vcm.pressure_at_interface
   .. autofunction:: vcm.surface_pressure_from_delp

Geopotential height
-------------------

   .. autofunction:: vcm.height_at_midpoint
   .. autofunction:: vcm.height_at_interface

Atmospheric layer thickness
---------------------------

   .. autofunction:: vcm.calc.thermo.vertically_dependent.hydrostatic_dz
   .. autofunction:: vcm.density
   .. autofunction:: vcm.pressure_thickness
   .. autofunction:: vcm.layer_mass

Surface geopotential
--------------------

   .. autofunction:: vcm.calc.thermo.vertically_dependent.dz_and_top_to_phis

Heat
----

   .. autofunction:: vcm.net_heating
   .. autofunction:: vcm.column_integrated_heating_from_isobaric_transition
   .. autofunction:: vcm.column_integrated_heating_from_isochoric_transition
   .. autofunction:: vcm.calc.thermo.local.liquid_ice_temperature
   .. autofunction:: vcm.calc.thermo.local.latent_heat_vaporization
   .. autofunction:: vcm.potential_temperature
   .. autofunction:: vcm.internal_energy

Moisture
--------

   .. autofunction:: vcm.latent_heat_flux_to_evaporation
   .. autofunction:: vcm.calc.thermo.local.surface_evaporation_mm_day_from_latent_heat_flux
   .. autofunction:: vcm.net_precipitation
   .. autofunction:: vcm.minus_column_integrated_moistening
   .. autofunction:: vcm.calc.thermo.local.total_water
   .. autofunction:: vcm.calc.thermo.vertically_dependent.column_integrated_liquid_water_equivalent
   .. autofunction:: vcm.saturation_pressure
   .. autofunction:: vcm.relative_humidity
   .. autofunction:: vcm.specific_humidity_from_rh