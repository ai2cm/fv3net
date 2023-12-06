import cftime
import dataclasses
import logging
import numpy as np
import pandas as pd
import xarray as xr
import mpi4py.MPI as MPI
from datetime import timedelta
from typing import (
    Optional,
    MutableMapping,
    Hashable,
    Mapping,
    cast,
    Sequence,
    Dict,
    Union,
    Any,
)

import pace.util
from pace.util import constants
from pace.util.communicator import Quantity, array_buffer
import fv3fit
from fv3fit._shared import get_dir
from fv3fit._shared.halos import append_halos_using_mpi, append_halos
from fv3fit.reservoir.adapters import ReservoirDatasetAdapter
from runtime.names import SST, TSFC, MASK, SPHUM, TEMP
from runtime.tendency import add_tendency, tendencies_from_state_updates
from runtime.diagnostics import (
    enforce_heating_and_moistening_tendency_constraints,
    compute_diagnostics,
)
from .prescriber import sst_update_from_reference
from .machine_learning import rename_dataset_members, NameDict
from ..scatter import (
    scatter_within_tile,
    gather_from_subtiles,
    scatter_global,
    gather_global,
)

GLOBAL_COMM = MPI.COMM_WORLD
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReservoirConfig:
    """
    Reservoir model configuration.

    Attributes:
        models: Mapping from rank to reservoir model path to load
        synchronize_steps: Number of steps to synchronize the reservoir
            before prediction
        reservoir_timestep: Timestep of the reservoir model
        time_average_inputs: Whether to time average inputs to the reservoir
            increment and prediction (if hybrid) steps.  Uses running averages
            to match the reservoir timestep.
        diagnostic_only: Whether to run the reservoir in diagnostic mode (no
            state updates)
        warm_start: Whether to use the saved state from a pre-synced reservoir
        rename_mapping: mapping from field names used in the underlying
            reservoir model to names used in fv3gfs wrapper
        hydrostatic (optional): whether simulation is hydrostatic.
            For net heating diagnostic. Defaults to false.
        mse_conserving_limiter (optional): whether to use MSE-conserving humidity
            limiter. Defaults to false.
        incrementer_offset (optional): time offset to control when the increment
            step is called.  Useful for delaying the increment until time averaged
            inputs are available.
        reservoir_input_offset (optional): time offset to control when
            requested variables are stored to use in the increment step
    """

    models: Mapping[Union[int, str], str]
    synchronize_steps: int = 1
    reservoir_timestep: str = "3h"  # TODO: Could this be inferred?
    time_average_inputs: bool = False
    diagnostic_only: bool = False
    warm_start: bool = False
    rename_mapping: NameDict = dataclasses.field(default_factory=dict)
    hydrostatic: bool = False
    mse_conserving_limiter: bool = False
    incrementer_offset: Optional[str] = None
    reservoir_input_offset: Optional[Mapping[str, str]] = None

    def __post_init__(self):
        # This handles cases in automatic config writing where json/yaml
        # do not allow integer keys
        _models = {}
        for key, url in self.models.items():
            try:
                int_key = int(key)
                _models[int_key] = url
            except (ValueError) as e:
                raise ValueError(
                    "Keys in reservoir_corrector.models must be integers "
                    "or string representation of integers."
                ) from e
        self.models = _models


class _FiniteStateMachine:
    """
    A simple state machine to keep to a shared state between the increment
    and predict steppers that are separated across the time loop.
    """

    INCREMENT = "increment"
    PREDICT = "predict"

    def __init__(self) -> None:
        self._last_called = None
        self._num_increments_completed = 0

    @property
    def completed_increments(self):
        return self._num_increments_completed

    def to_incremented(self):
        # incrementing allowed anytime, e.g., synchronizing
        self._last_called = self.INCREMENT
        self._num_increments_completed += 1

    def to_predicted(self):
        # predict only allowed after increment has been called
        if self._last_called != self.INCREMENT:
            raise ValueError("Must call increment before next prediction")
        self._last_called = self.PREDICT

    def __call__(self, state: str):
        if state == self.INCREMENT:
            self.to_incremented()
        elif state == self.PREDICT:
            self.to_predicted()
        else:
            raise ValueError(
                f"Unknown state provided to _ReservoirStepperState {state}"
            )


class TimeAverageInputs:
    """
    Copy of time averaging components from runtime.diagnostics.manager to
    use for averaging inputs to the reservoir model.
    """

    def __init__(self, variables: Sequence[str]):
        self.variables = variables
        self._running_total: Dict[str, xr.DataArray] = {}
        self._n: Dict[str, int] = {}
        self._recorded_units: Dict[str, str] = {}

    def increment_running_average(self, inputs: Mapping[str, xr.DataArray]):
        for key in inputs:
            self._recorded_units[key] = inputs[key].attrs.get("units", "unknown")

        for key in self.variables:
            if key in self._running_total:
                self._running_total[key] += inputs[key]
                self._n[key] += 1
            else:
                self._running_total[key] = inputs[key].copy()
                self._n[key] = 1

    def _reset_running_average(self, key: str):
        del self._running_total[key]
        del self._n[key]

    def get_average(self, key: str):
        if key not in self.variables or key not in self._running_total:
            raise ValueError(
                f"Variable {key} not present in time averaged inputs"
                f" {self._running_total.keys()} [set: {self.variables}]"
            )

        avg = self._running_total[key] / self._n[key]
        avg.attrs["units"] = self._recorded_units[key]

        self._reset_running_average(key)
        logger.info(f"Retrieved time averaged input data for reservoir: {key}")

        return avg

    def get_averages(self):

        averages = {k: self.get_average(k) for k in self.variables}

        logger.info(
            "Retrieved all time averaged input data for reservoir:"
            f" {averages.keys()}"
        )

        return averages


def _scatter_stepper_return(communicator, tendencies, diags, state):

    tendencies = scatter_within_tile(communicator, tendencies)
    diags = scatter_within_tile(communicator, diags)
    state = scatter_within_tile(communicator, state)

    tendencies = tendencies if tendencies else {}
    diags = diags if diags else {}
    state = state if state else {}

    return tendencies, diags, state


class FullTileScatterComm(pace.util.CubedSphereCommunicator):
    @classmethod
    def from_cubed_sphere_communicator(cls, communicator):
        return cls(
            communicator.comm,
            communicator.partitioner,
            force_cpu=communicator._force_cpu,
            timer=communicator.timer,
        )

    def scatter(
        self,
        send_quantity: Optional[Quantity] = None,
        recv_quantity: Optional[Quantity] = None,
    ) -> Quantity:
        """
        Transfer a whole tiles from a global cubedsphere to each
        tile root rank.

        Args:
            send_quantity: quantity to send, only required/used on the root rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.ROOT_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the root rank")
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            metadata = self.comm.bcast(send_quantity.metadata, root=constants.ROOT_RANK)
        else:
            metadata = self.comm.bcast(None, root=constants.ROOT_RANK)
        shape = metadata.extent[1:]
        if recv_quantity is None:
            recv_quantity = self._get_scatter_recv_quantity(shape, metadata)

        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            total_ranks = self.partitioner.total_ranks
            with array_buffer(
                self._maybe_force_cpu(metadata.np).zeros,
                (total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for i in range(0, self.partitioner.total_ranks):
                    tile = self.partitioner.tile_index(i)
                    sendbuf.assign_from(
                        send_quantity.view[tile], buffer_slice=np.index_exp[i, :],
                    )
                self._Scatter(
                    metadata.np,
                    sendbuf.array,
                    recv_quantity.view[:],
                    root=constants.ROOT_RANK,
                )
        else:
            self._Scatter(
                metadata.np, None, recv_quantity.view[:], root=constants.ROOT_RANK,
            )
        return recv_quantity


class _ReservoirStepper:

    label = "base_reservoir_stepper"

    def __init__(
        self,
        model: ReservoirDatasetAdapter,
        init_time: cftime.DatetimeJulian,
        reservoir_timestep: timedelta,
        model_timestep: float,
        synchronize_steps: int,
        state_machine: Optional[_FiniteStateMachine] = None,
        diagnostic_only: bool = False,
        input_averager: Optional[TimeAverageInputs] = None,
        rename_mapping: Optional[NameDict] = None,
        warm_start: bool = False,
        communicator: Optional[pace.util.CubedSphereCommunicator] = None,
        required_variables: Optional[Sequence[str]] = None,
        hydrostatic: bool = False,
        mse_conserving_limiter: bool = False,
        incrementer_offset: Optional[timedelta] = None,
        reservoir_input_offset: Optional[Mapping[str, timedelta]] = None,
    ):
        self.model = model
        self.synchronize_steps = synchronize_steps
        self.initial_time = init_time
        self.timestep = reservoir_timestep
        self.model_timestep = model_timestep
        self.is_diagnostic = diagnostic_only
        self.input_averager = input_averager
        self.communicator = communicator
        self.warm_start = warm_start
        self._required_variables = required_variables
        self.hydrostatic = hydrostatic
        self.mse_conserving_limiter = mse_conserving_limiter
        self._incrementer_offset = (
            incrementer_offset if incrementer_offset is not None else timedelta(0)
        )
        self._reservoir_input_offset = (
            reservoir_input_offset if reservoir_input_offset is not None else {}
        )

        if state_machine is None:
            state_machine = _FiniteStateMachine()
        self._state_machine = state_machine

        if self.warm_start:
            if self.synchronize_steps != 0:
                raise ValueError(
                    "Warm start specified with non-zero sync steps.  Ensure that"
                    " the reservoir model is pre-synchronized and set sync steps to 0"
                    " in the configuration."
                )

            # allows for immediate predict
            self._state_machine(self._state_machine.INCREMENT)

        if rename_mapping is None:
            rename_mapping = cast(NameDict, {})
        self.rename_mapping = rename_mapping

        # storage for intermediate states while incrementing
        self._intermediate_storage: Mapping[str, Any] = {}

    @property
    def completed_sync_steps(self):
        return self._state_machine.completed_increments

    def __call__(self, time, state):
        raise NotImplementedError(
            "Must use a wrapper Stepper for the reservoir to use in the TimeLoop"
        )

    def _is_rc_update_step(self, time):
        remainder = (time - self.initial_time) % self.timestep
        return remainder == timedelta(0)

    def get_diagnostics(self, state, tendency):
        diags: MutableMapping[Hashable, xr.DataArray] = {}
        return diags, xr.DataArray()

    def _retrieve_fv3_state(self, state, reservoir_variables):
        """Return state mapping w/ fv3gfs state variable names"""
        state_variables = [self.rename_mapping.get(k, k) for k in reservoir_variables]
        return xr.Dataset({k: state[k] for k in state_variables})

    def _rename_inputs_for_reservoir(self, inputs):
        """
        Adjust collected fv3gfs state from original variable names
        to reservoir names
        """
        state_to_reservoir_names = {v: k for k, v in self.rename_mapping.items()}
        return xr.Dataset(
            {state_to_reservoir_names.get(k, k): inputs[k] for k in inputs}
        )


class ReservoirIncrementOnlyStepper(_ReservoirStepper):
    """
    Stepper that only increments the state of the reservoir.  Useful because we
    need to call this using the input of the model prior to any updates from other
    steppers.  The model adapter should be the same adapter provided to the
    ReservoirStepper.
    """

    label = "reservoir_incrementer"

    @property
    def n_halo_points(self):
        return self.model.input_overlap

    def _append_halos_mpi(self, inputs):
        """
        Append halos to inputs using mpi4py.
        """
        n_halo_points = self.model.input_overlap
        if n_halo_points > 0:
            try:
                rc_in_with_halos = append_halos_using_mpi(inputs, n_halo_points)
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state fields"
                    " during reservoir increment update"
                )
            inputs = rc_in_with_halos
        return inputs

    def _append_halos_global(self, inputs):
        if self.communicator is None:
            raise ValueError("Cannot append global halos without communicator")

        logger.info(
            f"appending halo rank {self.communicator.rank}, "
            f"original input {str(inputs)}"
        )
        global_ds = gather_global(self.communicator, inputs)

        if self.communicator.rank == 0:
            with_halos = append_halos(global_ds, self.model.input_overlap)
        else:
            with_halos = None

        scatter_comm = FullTileScatterComm.from_cubed_sphere_communicator(
            self.communicator
        )
        tile_with_halo = scatter_global(scatter_comm, with_halos)
        return tile_with_halo

    def _get_inputs_from_state(self, state):
        """
        Get all required inputs for incrementing w/ halos

        Add the slmask if SST is an input variable for masking
        """
        if self._required_variables is None:
            variables = self.model.nonhybrid_input_variables
        else:
            variables = self._required_variables

        state_inputs = self._retrieve_fv3_state(state, variables)

        if self.communicator and self.n_halo_points > 0:
            reservoir_inputs = self._append_halos_global(state_inputs)
        elif self.communicator and self.n_halo_points == 0:
            reservoir_inputs = gather_from_subtiles(self.communicator, state_inputs)
        elif self.communicator is None and self.n_halo_points > 0:
            reservoir_inputs = self._append_halos_mpi(state_inputs)

        reservoir_inputs = self._rename_inputs_for_reservoir(state_inputs)

        return reservoir_inputs

    def increment_reservoir(self, inputs):
        """Should be called at beginning of time loop"""

        if self.completed_sync_steps == 0 and not self.warm_start:
            self.model.reset_state()
        self._state_machine(self._state_machine.INCREMENT)
        self.model.increment_state(inputs)

    def _store_inputs_for_increment(self, time, inputs):
        """
        Store a given input for use with the increment
        """

        for key, data in inputs.items():
            offset = self._reservoir_input_offset.get(key, timedelta(0))
            if self._is_rc_update_step(time + offset):
                logger.info(f"Storing reservoir input {key} for increment: time {time}")
                to_store = data
                if self.input_averager is not None and key != "sst":
                    # TODO: if this works, make configurable
                    # hack to keep at instantaneous SST (which is weekly for RC)
                    to_store = self.input_averager.get_average(key)
                self._intermediate_storage[key] = to_store

    def _get_inputs_for_increment(self):
        inputs = xr.Dataset({**self._intermediate_storage})
        self._intermediate_storage = {}
        return inputs

    def __call__(self, time, state):

        diags = {}
        tendencies = {}
        output_state = {}

        # add to averages
        inputs = self._get_inputs_from_state(state)
        if self.input_averager is not None:
            self.input_averager.increment_running_average(inputs)

        self._store_inputs_for_increment(time, inputs)

        # Add a call to a store for state if offset time is reached
        # Take the averager update out of the is _rc_update_step
        # adjust the time such that the increment update happens
        # at the correct tiem to gather all the inputs

        if self._is_rc_update_step(time + self._incrementer_offset):

            inputs = self._get_inputs_for_increment()

            logger.info(f"Incrementing rc at time {time}")
            self.increment_reservoir(inputs)
            diags = rename_dataset_members(
                inputs, {k: f"{self.rename_mapping.get(k, k)}_rc_in" for k in inputs}
            )

            # prevent conflict with non-halo diagnostics
            if self.model.input_overlap > 0:
                overlap = self.model.input_overlap
                isel_kwargs = {
                    dim: slice(overlap, -overlap)
                    for dim in diags.dims
                    if dim in ["x", "y"]
                }
                diags = diags.isel(**isel_kwargs)

            if self.communicator:
                logger.info(
                    f"Scattering increment diags (rank {GLOBAL_COMM.Get_rank()}):"
                    f" {list(diags.keys())}"
                )
                tendencies, diags, output_state = _scatter_stepper_return(
                    self.communicator, tendencies, diags, output_state
                )

        return tendencies, diags, output_state


class ReservoirPredictStepper(_ReservoirStepper):
    """
    Stepper that predicts using the current state of the reservoir.  Meant to
    be called after ReservoirIncrementOnlyStepper has been called but it's left
    up to the caller to ensure these occur in the right order.
    """

    label = "reservoir_predictor"
    DIAGS_OUTPUT_SUFFIX = "rc_out"

    def predict(self, inputs, pre_predict_state):
        """Called at the end of timeloop after time has ticked from t -> t+1"""

        self._state_machine(self._state_machine.PREDICT)
        result = self.model.predict(inputs)
        output_state = rename_dataset_members(result, self.rename_mapping)

        diags = rename_dataset_members(
            output_state, {k: f"{k}_{self.DIAGS_OUTPUT_SUFFIX}" for k in output_state}
        )

        for k, v in output_state.items():
            v.attrs["units"] = pre_predict_state[k].attrs.get("units", "unknown")

        # no halo necessary for potential hybrid inputs
        # +1 to align with the necessary increment before any prediction
        if (
            self._state_machine.completed_increments <= self.synchronize_steps
            or self.is_diagnostic
        ):
            output_state = {}

        if SST in output_state:
            # note that refrence to update from is the predicted state here
            sst_updates = sst_update_from_reference(
                pre_predict_state, output_state, reference_sst_name=SST
            )
            output_state.update(sst_updates)

        return {}, diags, output_state

    def __call__(self, time, state):

        # won't evaluate to true until we've reached the step before the next increment
        # e.g., if fv3 has k timesteps between rc timestep, on t + k - 1, the timestep
        # at the end will have ticked over to t + k in the middle of the called wrapper
        # steps prior to predict, we'll maybe use the integrated
        # hybrid quantites from t -> t + k, make the rc prediction for t + k, and then
        # increment during the next time loop based on those outputs.

        # Need to gather TSFC and SST for update_from_reference, which complicates
        # the gather requirements.  Otherwise those fields are subdomains.
        if self._required_variables is not None:
            use_variables = self._required_variables
        elif self.model.is_hybrid:
            use_variables = list(self.model.hybrid_variables)
        else:
            use_variables = []

        retrieved_state = self._retrieve_fv3_state(state, use_variables)
        if self.communicator and use_variables:
            logger.info(
                f"gathering predictor state (rank: {GLOBAL_COMM.Get_rank()}):"
                f" {list(retrieved_state.keys())}"
            )
            retrieved_state = gather_from_subtiles(self.communicator, retrieved_state)

        if self.model.is_hybrid:
            hybrid_inputs = self._rename_inputs_for_reservoir(retrieved_state)
            hybrid_inputs = hybrid_inputs[[k for k in self.model.hybrid_variables]]
        else:
            hybrid_inputs = xr.Dataset()

        if self.input_averager is not None:
            self.input_averager.increment_running_average(hybrid_inputs)

        if self._is_rc_update_step(time):
            logger.info(f"Reservoir model predict at time {time}")
            if self.input_averager is not None:
                hybrid_inputs.update(self.input_averager.get_averages())
            tendencies, diags, output_state = self.predict(
                hybrid_inputs, retrieved_state
            )

            hybrid_diags = rename_dataset_members(
                hybrid_inputs,
                {k: f"{self.rename_mapping.get(k, k)}_hyb_in" for k in hybrid_inputs},
            )
            diags.update(hybrid_diags)

            if self.communicator:
                logger.info(
                    f"Scattering predict return values (rank {GLOBAL_COMM.Get_rank()}):"
                    f" {list(output_state.keys()) + list(diags.keys())}"
                )
                tendencies, diags, output_state = _scatter_stepper_return(
                    self.communicator, tendencies, diags, output_state
                )

            # This check is done on the _rc_out diags since those are always available.
            # This allows zero field diags to be returned on timesteps where the
            # reservoir is not updating the state.
            diags_Tq_vars = {f"{v}_{self.DIAGS_OUTPUT_SUFFIX}" for v in [TEMP, SPHUM]}

            if diags_Tq_vars.issubset(list(diags.keys())):
                # TODO: Currently the reservoir only predicts updated states and returns
                # empty tendencies. If tendency predictions are implemented in the
                # prognostic run, the limiter/conservation updates should be updated to
                # take this option into account and use predicted tendencies directly.
                tendencies_from_state_prediction = tendencies_from_state_updates(
                    initial_state=state,
                    updated_state=output_state,
                    dt=self.model_timestep,
                )
                (
                    tendency_updates_from_constraints,
                    diagnostics_updates_from_constraints,
                ) = enforce_heating_and_moistening_tendency_constraints(
                    state=state,
                    tendency=tendencies_from_state_prediction,
                    timestep=self.model_timestep,
                    mse_conserving=self.mse_conserving_limiter,
                    hydrostatic=self.hydrostatic,
                    temperature_tendency_name="dQ1",
                    humidity_tendency_name="dQ2",
                    zero_fill_missing_tendencies=True,
                )

                diags.update(diagnostics_updates_from_constraints)
                output_state = add_tendency(
                    state=state,
                    tendencies=tendency_updates_from_constraints,
                    dt=self.model_timestep,
                )
                tendencies.update(tendency_updates_from_constraints)
        else:
            tendencies, diags, output_state = {}, {}, {}

        return tendencies, diags, output_state

    def get_diagnostics(self, state, tendency):
        diags = compute_diagnostics(state, tendency, self.label, self.hydrostatic)
        return diags, diags[f"net_moistening_due_to_{self.label}"]


class _GatherScatterStateStepper:
    """
    A class that retrieves specific state variables from subtiles and
    gathers them to the root rank. Then updates state based on scattered
    state from the root reservoir prediction.
    """

    def __init__(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        variables: Sequence[str],
        initial_time: cftime.DatetimeJulian,
        reservoir_timestep: timedelta,
        offset: timedelta = timedelta(0),
        extra_gather_scatter: bool = False,
    ) -> None:
        self.initial_time = initial_time
        self.timestep = reservoir_timestep
        self.communicator = communicator
        self.variables = variables if variables is not None else []
        self.offset = offset
        self.is_diagnostic = False
        self.halo_gather_scatter = extra_gather_scatter

    def __call__(self, time, state):

        output_state = {}
        tendencies = {}
        diags = {}

        rank = GLOBAL_COMM.Get_rank()
        retrieved_state = xr.Dataset({k: state[k] for k in self.variables})
        logger.info(
            f"Gathering from gs obj at time {time}, rank({rank}),"
            f" {list(retrieved_state.keys())}"
        )

        if self.halo_gather_scatter:
            gather_global(self.communicator, retrieved_state)
            scatter_comm = FullTileScatterComm.from_cubed_sphere_communicator(
                self.communicator
            )
            scatter_global(scatter_comm, xr.Dataset())
        else:
            gather_from_subtiles(self.communicator, retrieved_state)

        if self._is_rc_update_step(time + self.offset):

            logger.info(
                f"GS obj scatter (rank {rank}):"
                f" {list(output_state.keys()) + list(diags.keys())}"
            )
            tendencies, diags, output_state = _scatter_stepper_return(
                self.communicator, tendencies, diags, output_state
            )

        return tendencies, diags, output_state

    def _is_rc_update_step(self, time):
        remainder = (time - self.initial_time) % self.timestep
        return remainder == timedelta(0)

    def get_diagnostics(self, state, tendency):
        diags: MutableMapping[Hashable, xr.DataArray] = {}
        return diags, xr.DataArray()


def open_rc_model(path: str) -> ReservoirDatasetAdapter:
    with get_dir(path) as f:
        model = cast(ReservoirDatasetAdapter, fv3fit.load(f))
    return model


def _get_time_averagers(model, do_time_average):
    if do_time_average:
        increment_averager = TimeAverageInputs(model.model.input_variables)
        predict_averager: Optional[TimeAverageInputs]
        if model.is_hybrid:
            hybrid_inputs = model.hybrid_variables
            variables = hybrid_inputs if hybrid_inputs is not None else []
            predict_averager = TimeAverageInputs(variables)
        else:
            predict_averager = None
    else:
        increment_averager, predict_averager = None, None

    return increment_averager, predict_averager


def _get_reservoir_steppers(
    model,
    config: ReservoirConfig,
    init_time: cftime.DatetimeJulian,
    model_timestep: float,
    incrementer_offset: Optional[timedelta] = None,
    communicator=None,
    increment_variables=None,
    predictor_variables=None,
):

    state_machine = _FiniteStateMachine()
    rc_tdelta = pd.to_timedelta(config.reservoir_timestep)
    increment_averager, predict_averager = _get_time_averagers(
        model, config.time_average_inputs
    )

    reservoir_input_offset = None
    if config.reservoir_input_offset is not None:
        reservoir_input_offset = {
            k: pd.to_timedelta(v) for k, v in config.reservoir_input_offset.items()
        }

    incrementer = ReservoirIncrementOnlyStepper(
        model,
        init_time,
        reservoir_timestep=rc_tdelta,
        synchronize_steps=config.synchronize_steps,
        state_machine=state_machine,
        input_averager=increment_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
        communicator=communicator,
        required_variables=increment_variables,
        model_timestep=model_timestep,
        incrementer_offset=incrementer_offset,
        reservoir_input_offset=reservoir_input_offset,
    )
    predictor = ReservoirPredictStepper(
        model,
        init_time,
        reservoir_timestep=rc_tdelta,
        synchronize_steps=config.synchronize_steps,
        state_machine=state_machine,
        diagnostic_only=config.diagnostic_only,
        input_averager=predict_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
        communicator=communicator,
        required_variables=predictor_variables,
        model_timestep=model_timestep,
        hydrostatic=config.hydrostatic,
        mse_conserving_limiter=config.mse_conserving_limiter,
    )
    return incrementer, predictor


def _more_ranks_than_models(num_models: int, num_ranks: int):
    if num_models > num_ranks:
        raise ValueError(
            f"Number of models provided ({num_models}) is greater than"
            f"the number of ranks ({num_ranks})."
        )
    elif num_models < num_ranks:
        if num_ranks % num_models != 0:
            raise ValueError(
                f"Number of ranks ({num_ranks}) must be divisible by"
                f"the number of models ({num_models})."
            )
        return True
    else:
        return False


def _initialize_steppers_for_gather_scatter(
    model,
    config,
    init_time,
    model_timestep,
    rank,
    tile_root,
    communicator,
    incrementer_offset,
    halo_gather_scatter,
):

    if rank == 0:
        variables = [
            config.rename_mapping.get(k, k) for k in model.nonhybrid_input_variables
        ]
        if model.is_hybrid:
            predictor_variables = [
                config.rename_mapping.get(k, k) for k in model.hybrid_variables
            ]
        else:
            predictor_variables = []

        if SST in [config.rename_mapping.get(k, k) for k in model.output_variables]:
            predictor_variables += [SST, TSFC, MASK]
    else:
        variables = None
        predictor_variables = None

    variables = GLOBAL_COMM.bcast(variables, root=0)
    predictor_variables = GLOBAL_COMM.bcast(predictor_variables, root=0)

    if rank != tile_root:
        logging.info(f"Getting gather/scatter steppers for rank {rank}")
        timestep = pd.to_timedelta(config.reservoir_timestep)
        incrementer = _GatherScatterStateStepper(
            communicator,
            variables,
            init_time,
            timestep,
            offset=incrementer_offset,
            extra_gather_scatter=halo_gather_scatter,
        )
        predictor = _GatherScatterStateStepper(
            communicator, predictor_variables, init_time, timestep
        )
    else:
        logging.info(f"Getting main steppers for rank {rank}")
        incrementer, predictor = _get_reservoir_steppers(
            model,
            config,
            init_time,
            model_timestep,
            incrementer_offset=incrementer_offset,
            communicator=communicator,
            increment_variables=variables,
            predictor_variables=predictor_variables,
        )

    return incrementer, predictor


def get_reservoir_steppers(
    config: ReservoirConfig,
    rank: int,
    init_time: cftime.DatetimeJulian,
    communicator: pace.util.CubedSphereCommunicator,
    model_timestep: float,
):
    """
    Gets both steppers needed by the time loop to increment the state using
    inputs from the beginning of the timestep and applying hybrid readout
    using the stepped underlying model + incremented RC state.

    Handles the situation where there are more ranks than models by creating
    gather/scatter steppers on ranks where there is no model to load.
    """
    logger.info(f"Getting steppers w/ init time: {init_time}")
    num_models = len(config.models)
    if _more_ranks_than_models(num_models, communicator.partitioner.total_ranks):
        tile_root = communicator.partitioner.tile_root_rank(rank)
        model_index = communicator.partitioner.tile_index(rank)
        require_scatter_gather = True
    else:
        tile_root = rank
        model_index = rank
        require_scatter_gather = False

    # used to add variables for SST masked update
    predictor_variables = None

    if rank == tile_root:
        logger.info(f"Loading reservoir model on rank {rank}")
        try:
            model = open_rc_model(config.models[model_index])
        except KeyError:
            raise KeyError(
                f"No reservoir model path found  for rank {rank}. "
                "Ensure that the rank key and model is present in the configuration."
            )
        if model.is_hybrid:
            predictor_variables = [
                config.rename_mapping.get(k, k)
                for k in model.hybrid_variables  # type: ignore
            ]
    else:
        model = None  # type: ignore

    if rank == 0:
        extra_gather_scatter = model.input_overlap > 0
        GLOBAL_COMM.bcast(extra_gather_scatter, root=0)
    else:
        extra_gather_scatter = GLOBAL_COMM.bcast(None, root=0)

    if config.incrementer_offset is not None:
        incrementer_offset = pd.to_timedelta(config.incrementer_offset)
    else:
        incrementer_offset = timedelta(seconds=0)

    if require_scatter_gather:
        incrementer, predictor = _initialize_steppers_for_gather_scatter(
            model,
            config,
            init_time,
            model_timestep,
            rank,
            tile_root,
            communicator,
            incrementer_offset,
            extra_gather_scatter,
        )
    else:
        if SST in [config.rename_mapping.get(k, k) for k in model.output_variables]:
            if predictor_variables is None:
                predictor_variables = []
            predictor_variables += [SST, TSFC, MASK]

        incrementer, predictor = _get_reservoir_steppers(
            model,
            config,
            init_time,
            model_timestep,
            incrementer_offset=incrementer_offset,
            predictor_variables=predictor_variables,
        )

    return incrementer, predictor
