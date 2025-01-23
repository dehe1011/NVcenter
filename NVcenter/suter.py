import numpy as np
import qutip as q

from . import CONST
from .helpers import adjust_space_dim, get_dipolar_matrix
from .spins import Spins


def H_Suter():
    """Calculates the Hamiltonian used in Hedge2020.

    Notes
    -----
        - We can ignore the upper left and lower right blocks because they connect states with an spin
        flip in the NV center (will not happen because of the big energy level splitting).
    """

    D = 2.87e9
    ve = -414e6
    vc = 0.158e6
    A_N = -2.16e6
    A_zz = -0.152e6
    A_zx = 0.110e6

    sz = q.sigmaz() * 0.5
    sx = q.sigmax() * 0.5
    sz_NV = q.Qobj(np.array([[0, 0], [0, -1]]))
    NV_energy_barrier = D * adjust_space_dim(2, sz_NV**2, 0) - (
        ve - A_N
    ) * adjust_space_dim(2, sz_NV, 0)
    H = (
        -vc * adjust_space_dim(2, sz, 1)
        + A_zz * q.tensor(sz_NV, sz)
        + A_zx * q.tensor(sz_NV, sx)
    )
    return H


def calc_hadamard_pulse_seq(C13_pos, suter_method=False):
    """Analytical way to prepare a superopsition state (not a gate!) assuming instananeous pulses
    as described in the Supplementary of Hedge2020.
    """

    register_config = [("NV", (0, 0, 0), 0, {}), ("C13", C13_pos, 0, {})]
    approx_level = "no_bath"
    spins = Spins(register_config, [], approx_level)

    spin1, spin2 = spins.register_spins
    dipolar_matrix = get_dipolar_matrix(
        spin1.spin_pos, spin2.spin_pos, spin1.gamma, spin2.gamma, suter_method=suter_method
    )
    A_zz = dipolar_matrix[2, 2]
    A_zx = dipolar_matrix[0, 2]
    ve = CONST["gamma_e"] * CONST["Bz"] / (2 * np.pi)
    vc = CONST["gamma_C"] * CONST["Bz"] / (2 * np.pi)

    v_minus = np.sqrt(A_zx**2 + (vc + A_zz) ** 2)
    k_minus = np.arctan2(A_zx, (A_zz + vc))

    T1 = 1 / (np.pi * v_minus) * np.arcsin(1 / (np.sqrt(2) * np.sin(k_minus)))
    T2 = 1 / (2 * np.pi * vc) * np.arccos((np.cos(k_minus) / np.sin(k_minus)))
    return T1, T2
