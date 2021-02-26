Times
=====

FV3GFS tags intermediate restart files with a timestamp of the form
%Y%m%d.%H%M%S.  Functions defined here are useful for working with those.  In
addition, the model uses a Julian calendar internally.  Most of the time this
means that a ``cftime.DatetimeJulian`` object is the most appropriate
representation of those times; however, sometimes it can be useful to cast those
to ``datetime.datetime`` objects.

Converting FV3 string timestamps to datetimes
---------------------------------------------

   .. automethod:: vcm.convenience.parse_datetime_from_str
   .. automethod:: vcm.convenience.convert_timestamps

Converting any kind of date to a datetime object 
------------------------------------------------

   .. automethod:: vcm.convenience.cast_to_datetime

Working with string timestamps
------------------------------

   .. automethod:: vcm.convenience.parse_timestep_str_from_path
   .. automethod:: vcm.convenience.shift_timestamp
   .. automethod:: vcm.convenience.encode_time

Rounding times
--------------

   .. automethod:: vcm.convenience.round_time