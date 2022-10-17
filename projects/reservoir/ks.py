from dataclasses import dataclass
from scipy.fftpack import fft, ifft, fftfreq
from numpy import pi, exp
import numpy as np
from typing import Optional


def integrate_ks_eqn(
    N, domain_size, ic, tmax, timestep,
):
    """
    This code is based on Kassam & Treferthen 2005
    doi: 10.1137/S1064827502410633


    """
    d = 1.0 * domain_size / (2.0 * pi * N)
    u = ic

    v = fft(u)

    # Precompute various ETDRK4 scalar quantities:
    h = timestep  # time step

    k = fftfreq(N, d=d)

    L = k ** 2 - k ** 4  # Fourier multipliers
    E = exp(h * L)
    E2 = exp(h * L / 2)
    M = 64  # no. of points for complex means
    r = exp(1j * pi * (np.arange(1, M + 1) - 0.5) / M)  # roots of unity
    LR = h * L[:, None] + r[None, :]

    Q = h * np.mean((exp(LR / 2) - 1) / LR, axis=1).real
    f1 = (
        h * np.mean((-4 - LR + exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1).real
    )
    f2 = h * np.mean((2 + LR + exp(LR) * (-2 + LR)) / LR ** 3, axis=1).real
    f3 = (
        h * np.mean((-4 - 3 * LR - LR ** 2 + exp(LR) * (4 - LR)) / LR ** 3, axis=1).real
    )

    # Main time-stepping loop:
    uu = [
        ic,
    ]

    nmax = round(tmax / h)

    g = -0.5j * k
    for n in range(nmax):
        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * fft(np.real(ifft(b)) ** 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * fft(np.real(ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        u = ifft(v).real
        uu.append(u)

    return np.array(uu)


def _generate_ic(input_dim, seed=0):
    np.random.seed(seed)
    x = 2 * pi * np.arange(1, input_dim + 1) / input_dim
    return (
        np.cos(np.random.uniform(-1, 1) * x)
        * (np.random.uniform(-1, 1) + np.sin(x))
        * (np.sin(np.random.uniform(-10, 10) * x))
    )


def generate_ks_time_series(
    input_size, domain_size, tmax, timestep=0.25, seed=0,
):
    ic = _generate_ic(input_size, seed)
    return integrate_ks_eqn(
        N=input_size, ic=ic, tmax=tmax, timestep=timestep, domain_size=domain_size
    )


@dataclass
class KSConfig:
    """ Generates 1D Kuramoto-Sivashinky solution time series
    N: number spatial points in final output vectors
    domain_size: periodic domain size
    timestep: timestep in solver
    spatial_downsampling_factor: optional, if >1 solver will run at this factor
        higher resolution and the final time series is downsampled to N.
        This is useful for stability.
    time_downsampling_factor: optional, same as above but for the time dimension.
    """

    N: int
    domain_size: int
    timestep: float
    spatial_downsampling_factor: int = 1
    time_downsampling_factor: int = 1
    subdomain_output_size: Optional[int] = None
    subdomain_overlap: Optional[int] = None
    subdomain_axis: int = 1

    def generate(self, n_steps: int, seed: int = 0):
        """
            n_steps: Number of timesteps to output. The interval between
                timesteps output is timestep * time_downsampling_factor
            seed: Random seed for initial condition

        Returns:
            Time series array with dims (time, x)
        """
        ks_time_series = generate_ks_time_series(
            input_size=self.N * self.spatial_downsampling_factor,
            tmax=n_steps * self.time_downsampling_factor,
            seed=seed,
            domain_size=self.domain_size,
            timestep=self.timestep,
        )

        ks_time_series = ks_time_series[
            slice(None, None, self.time_downsampling_factor), :
        ]
        ks_time_series = ks_time_series[
            :, slice(None, None, self.spatial_downsampling_factor)
        ]
        return ks_time_series
