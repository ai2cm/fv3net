import numpy as np

from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.model import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
)
from fv3fit.reservoir.reservoir import Reservoir, ReservoirHyperparameters
from fv3fit.reservoir.readout import ReservoirComputingReadout
from fv3fit.reservoir.transformers import DoNothingAutoencoder, TransformerGroup


def get_reservoir_computing_model(
    state_size=150,
    divider=RankXYDivider((2, 2), 0, rank_extent=(2, 2), z_feature_size=2),
    encoder=DoNothingAutoencoder([1, 1]),
    variables=("a", "b"),
    hybrid=False,
):

    input_size = divider.flat_subdomain_len
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    no_overlap_divider = divider.get_no_overlap_rank_divider()
    # multiplied by the number of subdomains since it's a combined readout

    if hybrid:
        coefs_feature_size = state_size + no_overlap_divider.flat_subdomain_len
    else:
        coefs_feature_size = state_size

    rng = np.random.RandomState(0)
    readout = ReservoirComputingReadout(
        coefficients=rng.randn(
            divider.n_subdomains,
            coefs_feature_size,
            no_overlap_divider.flat_subdomain_len,
        ),
        intercepts=rng.randn(
            divider.n_subdomains, no_overlap_divider.flat_subdomain_len
        ),
    )
    transformers = TransformerGroup(input=encoder, output=encoder, hybrid=encoder)
    if hybrid:
        no_overlap_divider = divider.get_no_overlap_rank_divider()
        input_mask = np.ones(
            (no_overlap_divider.n_subdomains, no_overlap_divider.flat_subdomain_len)
        )
        predictor = HybridReservoirComputingModel(
            input_variables=variables,
            output_variables=variables,
            hybrid_variables=variables,
            reservoir=reservoir,
            readout=readout,
            rank_divider=divider,
            transformers=transformers,
            hybrid_input_mask=input_mask,
        )
    else:
        predictor = ReservoirComputingModel(
            input_variables=variables,
            output_variables=variables,
            reservoir=reservoir,
            readout=readout,
            rank_divider=divider,
            transformers=transformers,
        )
    predictor.reset_state()

    return predictor
