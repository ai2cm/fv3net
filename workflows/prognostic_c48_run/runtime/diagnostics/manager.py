from typing import (
    Any,
    Sequence,
    Mapping,
    List,
    Union,
    Dict,
    Optional,
    MutableMapping,
)
from typing_extensions import Protocol

import cftime
import logging
import fv3gfs.util
import xarray as xr
import dataclasses

from .fortran import FortranFileConfig
from .time import TimeConfig, TimeContainer
from .tensorboard import TensorBoardSink

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DiagnosticFileConfig:
    """Configurations for Diagnostic Files

    Attributes:
        name: filename of a zarr to store the data in, e.g., 'diags.zarr'.
            Paths are relative to the run-directory root.
        variables: the variables to save. By default no diagnostics
            are stored. Example: ``["air_temperature", "cos_zenith_angle"]``.
        times: the time configuration
        chunks: mapping of dimension names to chunk sizes
        tensorboard: if True, then plot the variables and log them to a
            tensorflow summary directory at "<rundir>/tensorboard". These logs can
            then be viewed with tensorboard. If False (the default), then the
            variables are saved to zarr. Only supports 2D variables.
    """

    name: str = ""
    variables: Sequence[str] = dataclasses.field(default_factory=list)
    times: TimeConfig = dataclasses.field(default_factory=lambda: TimeConfig())
    chunks: Mapping[str, int] = dataclasses.field(default_factory=dict)
    tensorboard: bool = False

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def _get_sink(
        self, partitioner: fv3gfs.util.CubedSpherePartitioner, comm: Any,
    ) -> "Sink":
        if self.tensorboard:
            return TensorBoardSink()
        else:
            return ZarrSink(
                fv3gfs.util.ZarrMonitor(self.name, partitioner, mpi_comm=comm)
            )

    def diagnostic_file(
        self,
        initial_time: cftime.DatetimeJulian,
        partitioner: fv3gfs.util.CubedSpherePartitioner,
        comm: Any,
    ) -> "DiagnosticFile":

        return DiagnosticFile(
            variables=self.variables,
            times=self.times.time_container(initial_time),
            sink=self._get_sink(partitioner, comm),
        )


class Sink(Protocol):
    def sink(self, time: cftime.DatetimeJulian, data: Mapping[str, xr.DataArray]):
        pass


@dataclasses.dataclass
class ZarrSink:
    monitor: fv3gfs.util.ZarrMonitor

    def sink(self, time: cftime.DatetimeJulian, data: Mapping[str, xr.DataArray]):
        quantities = {
            # need units for from_data_array to work
            key: fv3gfs.util.Quantity.from_data_array(data[key])
            for key in data
        }

        # patch this in manually. the ZarrMonitor needs it.
        # We should probably modify this behavior.
        quantities["time"] = time
        self.monitor.store(quantities)


class DiagnosticFile:
    """A object representing a time averaged diagnostics file

    Replicates the abilities of the fortran models's diag_table by allowing
    the user to specify different output times for distinct sets of
    variables.

    Note:
        Outputting a snapshot is type of time-average (e.g. taking the average
        with respect to a point mass at a given time).

    """

    def __init__(
        self, variables: Sequence[str], times: TimeContainer, sink: Sink,
    ):
        """
        Note:

            The container used for times does not need to be a concrete list or python
            sequence. It only need to satisfy the abstract ``Container`` interface.
            Please see the special containers for outputting times above:

            - ``IntervalTimes``
            - ``SelectedTimes``

            as well as the generic ``All`` container that contains the entire
            Universe!
        """
        self.variables = variables
        self.times = times

        # variables used for averaging
        self._running_total: Dict[str, xr.DataArray] = {}
        self._current_label: Optional[cftime.DatetimeJulian] = None
        self._n = 0
        self._units: Dict[str, str] = {}
        self._sink = sink

    def observe(
        self, time: cftime.DatetimeJulian, diagnostics: Mapping[str, xr.DataArray]
    ):
        for key in diagnostics:
            self._units[key] = diagnostics[key].attrs.get("units", "unknown")

        label = self.times.indicator(time)
        if label is not None:
            if label != self._current_label:
                self.flush()
                self._reset_running_average(label, diagnostics)
            else:
                self._increment_running_average(diagnostics)

    def _reset_running_average(self, label, diagnostics):
        self._running_total = {
            key: val.copy() for key, val in diagnostics.items() if key in self.variables
        }
        self._current_label = label
        self._n = 1

    def _increment_running_average(self, diagnostics):
        self._n += 1
        for key in diagnostics:
            if key in self.variables:
                self._running_total[key] += diagnostics[key]

    def flush(self):
        if self._current_label is not None:
            average = {key: val / self._n for key, val in self._running_total.items()}
            for key in average:
                average[key].attrs["units"] = self._units[key]
            data_to_sink = {
                key: average[key] for key in average if key in self.variables
            }
            self._sink.sink(self._current_label, data_to_sink)

    def __del__(self):
        self.flush()


def get_diagnostic_files(
    configs: Sequence[DiagnosticFileConfig],
    partitioner: fv3gfs.util.CubedSpherePartitioner,
    comm: Any,
    initial_time: cftime.DatetimeJulian,
) -> List[DiagnosticFile]:
    """Initialize a list of diagnostic file objects from a configuration dictionary
    Note- the default here is to save no variables in the diagnostics. This
    can be overwritten by inserting a default diagnostics
    config entry for each runfile, e.g. ../prepare_config.py does this for
    the sklearn runfile.

    Args:
        configs: A sequence of DiagnosticFileConfigs
        paritioner: a partioner object used for writing, maybe it would be
            cleaner to pass a factory
        comm: an MPI Comm object
        initial_time: the initial time of the simulation.

    """
    return [
        config.diagnostic_file(initial_time, partitioner, comm) for config in configs
    ]


def get_chunks(
    diagnostic_file_configs: Sequence[Union[DiagnosticFileConfig, FortranFileConfig]],
) -> Mapping[str, Mapping[str, int]]:
    """Get a mapping of diagnostic file name to chunking from a sequence of diagnostic
    file configs."""
    chunks: MutableMapping = {}
    for diagnostic_file_config in diagnostic_file_configs:
        chunks[diagnostic_file_config.name] = diagnostic_file_config.chunks
    return chunks
