import numpy as np

from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.model import ReservoirComputingModel
from fv3fit.reservoir.reservoir import Reservoir, ReservoirHyperparameters
from fv3fit.reservoir.readout import ReservoirComputingReadout
from fv3fit.reservoir.transformers import DoNothingAutoencoder, TransformerGroup


def get_ReservoirComputingModel(
    state_size=150,
    rank_divider=RankXYDivider((2, 2), 0, rank_extent=(2, 2), z_feature_size=2),
    autoencoder=DoNothingAutoencoder([1, 1]),
    variables=("a", "b"),
):

    input_size = rank_divider.flat_subdomain_len
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(rank_divider.n_subdomains, state_size, input_size),
        intercepts=np.random.rand(input_size),
    )
    transformers = TransformerGroup(
        input=autoencoder, output=autoencoder, hybrid=autoencoder
    )
    predictor = ReservoirComputingModel(
        input_variables=variables,
        output_variables=variables,
        reservoir=reservoir,
        readout=readout,
        square_half_hidden_state=False,
        rank_divider=rank_divider,
        transformers=transformers,
    )

    return predictor
