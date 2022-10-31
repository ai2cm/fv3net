from dataclasses import dataclass
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from numpy import pi
import numpy as np
from typing import Optional

from fv3fit.reservoir.model import ImperfectModel

"""
The PDE solver code in integrate_ks_eqn is from
https://github.com/johnfgibson/julia-pde-benchmark/blob/master/codes/ksbenchmark.py

# License

Copyright (c) 2017 John F. Gibson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def integrate_ks_eqn(
    ic: np.ndarray, domain_size: int, dt: float, Nt: int, error_eps: float = 0
) -> np.ndarray:
    """
        Integrate the Kuramoto-Sivashinsky equation (Python)
        u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,domain_size], periodic BCs

        inputs:
        ic = initial condition (vector of u(x) values on uniform gridpoints))
        domain_size = domain length
        dt = time step
        Nt = number of integration timesteps
        error_eps = optional small error to add to make a more 'imperfect' solution
    outputs
        Time series of vectors u(x, nt*dt) at uniform x gridpoints
    """

    Nx = np.size(ic)
    kx = np.concatenate(
        (np.arange(0, Nx / 2), np.array([0]), np.arange(-Nx / 2 + 1, 0))
    )  # int wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2 * pi * kx / domain_size
    # real wavenumbers:    exp(alpha*x)
    D = 1j * alpha
    # D = d/dx operator in Fourier space
    L = pow(alpha, 2) - pow(alpha, 4)
    # linear operator -D^2 - D^3 in Fourier space
    G = -0.5 * D
    # -1/2 D operator in Fourier space

    # Express PDE as u_t = Lu + N(u), L is linear part, N nonlinear part.
    # Then Crank-Nicolson Adams-Bashforth discretization is
    #
    # (I - dt/2 L) u^{n+1} = (I + dt/2 L) u^n + 3dt/2 N^n - dt/2 N^{n-1}
    #
    # let A = (I - dt/2 L)
    #     B = (I + dt/2 L), then the CNAB timestep formula
    #
    # u^{n+1} = A^{-1} (B u^n + 3dt/2 N^n - dt/2 N^{n-1})

    # some convenience variables
    dt2 = dt / 2
    dt32 = 3 * dt / 2
    A = np.ones(Nx) + dt2 * L
    B = 1.0 / (np.ones(Nx) - dt2 * L)

    Nn = G * fft(
        ic * ic
    )  # compute -u u_x (spectral), notation Nn  = N^n     = N(u(n dt))
    Nn1 = Nn
    #                            notation Nn1 = N^{n-1} = N(u((n-1) dt))
    u = fft(ic)  # transform u (spectral)

    time_series = [
        ic,
    ]
    # timestepping loop
    for n in range(0, int(Nt)):

        Nn1 = Nn
        # shift nonlinear term in time: N^{n-1} <- N^n
        uu = np.real(ifft(u))
        uu = uu * uu
        uu = fft(uu)
        Nn = G * uu  # compute Nn == -u u_x (spectral)

        u = B * (A * u + dt32 * Nn - dt2 * Nn1 * (1 + error_eps))
        time_series.append(ifft(u))
    # For some reason, calling np.real within the for loop results in
    # list elements that still have complex parts with coeff 0.
    return np.real(np.array(time_series))


def _generate_ic(input_dim, seed=0):
    np.random.seed(seed)
    x = 2 * pi * np.arange(1, input_dim + 1) / input_dim
    return (
        np.cos(np.random.uniform(-1, 1) * x)
        * (np.random.uniform(-1, 1) + np.sin(x))
        * (np.sin(np.random.uniform(-10, 10) * x))
    )


def generate_ks_time_series(
    input_size, domain_size, n_steps, timestep=0.25, seed=0, error_eps=0.0
):
    ic = _generate_ic(input_size, seed)
    return integrate_ks_eqn(
        ic=ic, domain_size=domain_size, dt=timestep, Nt=n_steps, error_eps=error_eps
    )


@dataclass
class KuramotoSivashinskyConfig:
    """ Generates 1D Kuramoto-Sivashinsky solution time series
    N: number spatial points in final output vectors
    domain_size: periodic domain size
    timestep: solver timestep
    spatial_downsampling_factor: optional, if >1 solver will run at this factor
        higher resolution and the final time series is downsampled to N.
        This is useful for stability.
    """

    N: int
    domain_size: int
    timestep: float
    spatial_downsampling_factor: int = 1
    error_eps: float = 0.0
    subdomain_output_size: Optional[int] = None
    subdomain_overlap: Optional[int] = None
    subdomain_axis: int = 1

    def generate_from_seed(self, n_steps: int, seed: int = 0):
        """
            n_steps: Number of timesteps to output
            seed: Random seed for initial condition

        Returns:
            Time series array with dims (time, x)
        """
        ic = _generate_ic(self.N * self.spatial_downsampling_factor, seed=seed)
        ks_time_series = self.predict(n_steps=n_steps, initial_condition=ic)

        return ks_time_series

    def predict(self, n_steps: int, initial_condition: np.ndarray):
        if initial_condition.size != self.spatial_downsampling_factor * self.N:
            # upsample initial condition array by spatial_downsampling_factor
            ic_x = range(initial_condition.size)
            interp_ic = interp1d(ic_x, initial_condition)
            initial_condition = interp_ic(
                np.linspace(
                    ic_x[0], ic_x[-1], self.N * self.spatial_downsampling_factor
                )
            )
        ks_time_series = integrate_ks_eqn(
            ic=initial_condition,
            domain_size=self.domain_size,
            dt=self.timestep,
            Nt=n_steps,
            error_eps=self.error_eps,
        )
        return ks_time_series[:, :: self.spatial_downsampling_factor]


class ImperfectKSModel(ImperfectModel):
    def __init__(self, config: KuramotoSivashinskyConfig, reservoir_timestep: float):
        self.config = config
        self.reservoir_timestep = reservoir_timestep

        time_downsampling_factor = reservoir_timestep / config.timestep
        if not np.isclose(time_downsampling_factor, round(time_downsampling_factor)):
            raise ValueError(
                f"Reservoir timestep {reservoir_timestep} must be evenly divisble "
                f"by KS solver timestep {config.timestep}."
            )
        self.time_downsampling_factor = time_downsampling_factor

    def predict(self, ic):
        return integrate_ks_eqn(
            ic=ic,
            domain_size=self.config.domain_size,
            dt=self.config.timestep,
            Nt=self.time_downsampling_factor,
            error_eps=self.config.error_eps,
        )[-1]
