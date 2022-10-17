from dataclasses import dataclass
from scipy.fftpack import fft, ifft
from numpy import pi
import numpy as np
from typing import Optional


def integrate_ks_eqn(ic, domain_size, dt, Nt):
    """
        Code is from
        https://github.com/johnfgibson/julia-pde-benchmark/blob/master/codes/ksbenchmark.py,
        which uses the MIT license.

        Integrate the Kuramoto-Sivashinsky equation (Python)
        u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,domain_size], periodic BCs
    inputs
          ic = initial condition (vector of u(x) values on uniform gridpoints))
         domain_size = domain length
         dt = time step
         Nt = number of integration timesteps
      nsave = save every nsave-th time step

    outputs
          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints
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

        u = B * (A * u + dt32 * Nn - dt2 * Nn1)
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
    input_size, domain_size, n_steps, timestep=0.25, seed=0,
):
    ic = _generate_ic(input_size, seed)
    return integrate_ks_eqn(ic=ic, domain_size=domain_size, dt=timestep, Nt=n_steps)


@dataclass
class KSConfig:
    """ Generates 1D Kuramoto-Sivashinsky solution time series
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
            n_steps=n_steps * self.time_downsampling_factor,
            seed=seed,
            domain_size=self.domain_size,
            timestep=self.timestep,
        )

        ks_time_series = ks_time_series[:: self.time_downsampling_factor, :]
        ks_time_series = ks_time_series[:, :: self.spatial_downsampling_factor]

        return ks_time_series
