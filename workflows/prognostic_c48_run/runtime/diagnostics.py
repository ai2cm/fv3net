from typing import (
    Any,
    Sequence,
    Container,
    Mapping,
    List,
    Union,
    MutableMapping,
    Hashable,
)
import datetime
import cftime
import logging
from vcm.catalog import catalog
from vcm.cubedsphere import center_and_rotate_xy_winds
import fv3gfs.util

import xarray as xr

logger = logging.getLogger(__name__)

TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
cp = 1004
gravity = 9.81

State = MutableMapping[Hashable, xr.DataArray]


class All(Container):
    """A container that contains every thing
    
    This is useful for cases where we want an ``in`` check to always return True.

    Example:
        >>> all = All()
        >>> 'x' in all
        True
        >>> 1232.1 in all
        True
    """

    def __contains__(self, value: Any) -> bool:
        return True


class SelectedTimes(Container[cftime.DatetimeJulian]):
    TIME_FMT: str = r"%Y%m%d.%H%M%S"

    def __init__(self, times=Sequence[str]):
        self._time_stamps = times

        # see if there is an error
        self.times

    @property
    def _times(self) -> Sequence[datetime.datetime]:
        return [
            datetime.datetime.strptime(time, self.TIME_FMT)
            for time in self._time_stamps
        ]

    @property
    def times(self) -> Sequence[cftime.DatetimeJulian]:
        return [cftime.DatetimeJulian(*time.timetuple()) for time in self._times]

    def __contains__(self, time: cftime.DatetimeJulian) -> bool:
        return time in self.times


class IntervalTimes(Container[cftime.DatetimeJulian]):
    def __init__(
        self, frequency_seconds: Union[float, int], initial_time: cftime.DatetimeJulian,
    ):
        """
        Args:
            frequency_seconds: the output frequency from the initial time
            initial_time: the initial time to start the period
            
        """
        self._frequency_seconds = frequency_seconds
        self.initial_time = initial_time
        if self.frequency > datetime.timedelta(days=1.0) and initial_time is None:
            raise ValueError(
                "Minimum output frequency is daily when intial_time is not provided."
            )

    @property
    def frequency(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self._frequency_seconds)

    def __contains__(self, time) -> bool:
        time_since_initial_time = time - self.initial_time
        quotient = time_since_initial_time % self.frequency
        return quotient == datetime.timedelta(seconds=0)


def _assign_units_if_none_present(array: xr.DataArray, units=None):
    return array.assign_attrs(units=array.attrs.get("units", units))


class DiagnosticFile:
    """A object representing a diagnostics file

    Provides a similar interface as the "diag_table"

    Replicates the abilities of the fortran models's diag_table by allowing
    the user to specify different output times for distinct sets of
    variables.
    """

    def __init__(
        self,
        monitor: fv3gfs.util.ZarrMonitor,
        times: Container[cftime.DatetimeJulian],
        variables: Container,
    ):
        """
        Args:
            monitor: an underlying monitor to store the data in
            times: the set of times (potentially infinite) to save the data at
            variables: a container of variables to save

        Note:

            The containers used for times and variables do not need to be
            concrete lists or python sequences. They only need to satisfy the
            abstract ``Container`` interface. Please see the special
            containers for outputing times above:

            - ``IntervalTimes``
            - ``SelectedTimes``

            as well as the generic ``All`` container that contains the entire
            Universe!
        """
        self._monitor = monitor
        self.times = times
        self.variables = variables

    def observe(
        self, time: cftime.DatetimeJulian, diagnostics: Mapping[str, xr.DataArray]
    ):
        """Possibly store the data into the monitor
        """
        if time in self.times:
            quantities = {
                # need units for from_data_array to work
                key: fv3gfs.util.Quantity.from_data_array(
                    _assign_units_if_none_present(diagnostics[key], "unknown")
                )
                for key in diagnostics
                if key in self.variables
            }

            # patch this in manually. the ZarrMonitor needs it.
            # We should probably modify this behavior.
            quantities["time"] = time
            self._monitor.store(quantities)


def _get_times(
    d, initial_time: cftime.DatetimeJulian
) -> Container[cftime.DatetimeJulian]:
    kind = d.get("kind", "every")
    if kind == "interval":
        return IntervalTimes(d["frequency"], initial_time)
    elif kind == "selected":
        return SelectedTimes(d["times"])
    elif kind == "every":
        return All()
    else:
        raise NotImplementedError(f"Time {kind} not implemented.")


def _config_to_diagnostic_file(
    diag_file_config: Mapping, partitioner, comm, initial_time: cftime.DatetimeJulian,
) -> DiagnosticFile:
    monitor = fv3gfs.util.ZarrMonitor(
        diag_file_config["name"], partitioner, mpi_comm=comm
    )
    return DiagnosticFile(
        monitor=monitor,
        variables=diag_file_config.get("output_variables", All()),
        times=_get_times(diag_file_config.get("times", {}), initial_time),
    )


def get_diagnostic_files(
    config: Mapping,
    partitioner: fv3gfs.util.CubedSpherePartitioner,
    comm,
    initial_time: cftime.DatetimeJulian,
) -> List[DiagnosticFile]:
    """Initialize a list of diagnostic file objects from a configuration dictionary
    Note- the default here is to save all the variables in the diagnostics.
    The default set of variables can be overwritten by inserting a default diagnostics
    config entry for each runfile, e.g. ../prepare_config.py does this for
    the sklearn runfile.

    Args:
        config: A loaded "fv3config" dictionary with a "diagnostics" section
        paritioner: a partioner object used for writing, maybe it would be
            cleaner to pass a factory
        comm: an MPI Comm object
        initial_time: the initial time of the simulation.

    """
    diag_configs = config.get("diagnostics", [])
    if len(diag_configs) > 0:
        return [
            _config_to_diagnostic_file(config, partitioner, comm, initial_time)
            for config in diag_configs
        ]
    else:
        # Keep old behavior for backwards compatiblity
        output_name = config["scikit_learn"]["zarr_output"]
        default_config = {"name": output_name, "times": {}, "variables": All()}
        return [
            _config_to_diagnostic_file(default_config, partitioner, comm, initial_time)
        ]


def compute_ml_diagnostics(state, ml_tendency):

    net_moistening = (ml_tendency["dQ2"] * state[DELP] / gravity).sum("z")
    physics_precip = state[PRECIP_RATE]

    return dict(
        air_temperature=state[TEMP],
        specific_humidity=state[SPHUM],
        pressure_thickness_of_atmospheric_layer=state[DELP],
        net_moistening=(net_moistening)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated ML model moisture tendency"),
        net_heating=(ml_tendency["dQ1"] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated ML model heating"),
        water_vapor_path=(state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor"),
        physics_precip=(physics_precip)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        ),
    )


def compute_ml_momentum_diagnostics(state, tendency):
    delp = state[DELP]

    dQu = tendency.get("dQu", xr.zeros_like(delp))
    dQv = tendency.get("dQv", xr.zeros_like(delp))
    column_integrated_dQu = _mass_average(dQu, delp, "z")
    column_integrated_dQv = _mass_average(dQv, delp, "z")
    return dict(
        column_integrated_dQu=column_integrated_dQu.assign_attrs(
            units="m s^-2",
            description="column integrated zonal wind tendency due to ML",
        ),
        column_integrated_dQv=column_integrated_dQv.assign_attrs(
            units="m s^-2",
            description="column integrated meridional wind tendency due to ML",
        ),
    )


def rename_diagnostics(diags):
    """Postfix ML output names with _diagnostic and create zero-valued outputs in
    their stead. Function operates in place."""
    ml_tendencies = {
        "net_moistening",
        "net_heating",
        "column_integrated_dQu",
        "column_integrated_dQv",
    }
    ml_tendencies_in_diags = ml_tendencies & set(diags)
    for variable in ml_tendencies_in_diags:
        attrs = diags[variable].attrs
        diags[f"{variable}_diagnostic"] = diags[variable].assign_attrs(
            description=attrs["description"] + " (diagnostic only)"
        )
        diags[variable] = xr.zeros_like(diags[variable]).assign_attrs(attrs)


def compute_nudging_diagnostics(
    state: State, nudging_tendency: State, label: str = "_tendency_due_to_nudging"
):
    """
    Compute diagnostic variables for nudging"""

    net_moistening = (
        (nudging_tendency[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated moistening due to nudging")
    )
    net_heating = (
        (nudging_tendency[TEMP] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated heating due to nudging")
    )
    water_vapor_path = (
        (state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor")
    )
    physics_precip = (
        state[PRECIP_RATE]
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        )
    )

    diags = dict(
        net_moistening_due_to_nudging=net_moistening,
        net_heating_due_to_nudging=net_heating,
        water_vapor_path=water_vapor_path,
        physics_precip=physics_precip,
    )

    if ("x_wind" in nudging_tendency.keys()) and ("y_wind" in nudging_tendency.keys()):
        wind_rotation_matrix = catalog["wind_rotation/c48"].to_dask()
        u_tendency, v_tendency = center_and_rotate_xy_winds(
            wind_rotation_matrix, nudging_tendency["x_wind"], nudging_tendency["y_wind"]
        )
        rotation_mapping = {
            ("u-wind", "x_wind"): u_tendency,
            ("v-wind", "y_wind"): v_tendency,
        }
        for names, tendency in rotation_mapping.items():
            a_name, d_name = names
            integrated_wind_tendency = _mass_average(tendency, state[DELP], "z")
            diags[
                f"column_integrated_{a_name}_tendency_due_to_nudging"
            ] = integrated_wind_tendency.assign_attrs(
                units="m s^-2",
                description=(
                    f"column mass-averaged {a_name} wind tendency due to nudging"
                ),
            )

    if DELP in nudging_tendency:
        net_mass_tendency = (
            (nudging_tendency[DELP] / gravity)
            .sum("z")
            .assign_attrs(
                units="kg/m^2/s",
                description="column_integrated mass tendency due to nudging",
            )
        )
        diags["net_mass_tendency_due_to_nudging"] = net_mass_tendency
    diags.update(_append_key_label(nudging_tendency, label))

    return diags


def _append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


def _mass_average(da: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"):
    total_thickness = delp.sum(vertical_dim)
    mass_average = (da * delp).sum(vertical_dim) / total_thickness
    mass_average.assign_attrs(**da.attrs)
    return mass_average
