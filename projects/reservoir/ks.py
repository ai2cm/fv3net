from dataclasses import dataclass
from scipy.fftpack import fft, ifft, fftfreq
from numpy import pi, exp
import numpy as np
from typing import Optional


def integrate_ks_eqn(
    Ngrid, domain_size, ic, tmax, timestep,
):
    """
    This code is from
    kursiv.m - solution of Kuramoto-Sivashinsky equation by ETDRK4 scheme
    https://github.com/darlliu/follicle/blob/master/mtlab/kursiv.m

    u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
    computation is based on v = fft(u), so linear term is diagonal
    compare p27.m in Trefethen, "Spectral Methods in MATLAB", SIAM 2000
    AK Kassam and LN Trefethen, July 2002
    """
    tmax = tmax
    N = Ngrid
    d = 1.0 * domain_size / N
    u = ic

    v = fft(u)

    # Precompute various ETDRK4 scalar quantities:
    h = timestep  # time step

    nplt = 1

    k = fftfreq(N, d=d)

    L = k ** 2 - k ** 4  # Fourier multipliers
    E = exp(h * L)
    E2 = exp(h * L / 2)
    M = N  # no. of points for complex means
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
    uu = []
    tt = []

    nmax = round(tmax / h)

    g = -0.5j * k
    for n in range(nmax):
        t = n * h
        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * fft(np.real(ifft(b)) ** 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * fft(np.real(ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        if n % nplt == 0:
            u = ifft(v).real
            uu.append(u)
            tt.append(t)

    return uu


def _generate_ic(input_dim, seed=0):
    np.random.seed(seed)
    x = np.transpose(np.conj(np.arange(1, input_dim + 1))) / input_dim
    return (
        np.cos(np.random.uniform(-1, 1) * x * (2 * np.pi))
        * (np.random.uniform(-1, 1) + np.sin(x * (2 * np.pi)))
        * (np.sin(np.random.uniform(-10, 10) * x * (2 * np.pi)))
    )


def generate_ks_time_series(
    input_size, domain_size, tmax, timestep=0.25, seed=0,
):
    ic = _generate_ic(input_size, seed)
    return integrate_ks_eqn(
        Ngrid=input_size, ic=ic, tmax=tmax, timestep=timestep, domain_size=domain_size
    )


@dataclass
class KSConfig:
    """ Generates 1D Kuramoto-Shivashinky solution time series
    N: number spatial points in final output vectors
    domain_size: periodic domain size
    timestep: timestep in solver
    spatial_downsampling: optional, if >1 solver will run at a higher resolution
        and the final time series is downsampled to N. This is useful for stability.
    time_downsamplign: optional, same as above but for the time dimension.


    """

    N: int
    domain_size: int
    timestep: float
    spatial_downsampling: int = 1
    time_downsampling: int = 1
    subdomain_output_size: Optional[int] = None
    subdomain_overlap: Optional[int] = None
    subdomain_axis: int = 1

    def generate(self, n_steps: int, seed: int = 0):
        """
            n_steps: Number of timesteps to output. The interval between
                timesteps output is timestep * time_downsampling
            seed: Random seed for initial condition

        Returns:
            Time series array with dims (time, x)
        """
        ks_time_series = np.array(
            generate_ks_time_series(
                input_size=self.N * self.spatial_downsampling,
                tmax=n_steps * self.time_downsampling,
                seed=seed,
                domain_size=self.domain_size,
                timestep=self.timestep,
            )
        )
        ks_time_series = ks_time_series[slice(None, None, self.time_downsampling), :]
        ks_time_series = ks_time_series[:, slice(None, None, self.spatial_downsampling)]
        return ks_time_series