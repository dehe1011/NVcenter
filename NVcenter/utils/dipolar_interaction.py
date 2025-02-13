from itertools import product

import numpy as np
import scipy.constants as c
from scipy.optimize import fsolve

from .save_load import DEFAULTS
from .coordinates import spherical_to_cartesian

# -----------------------------------------------


def get_dipolar_matrix(pos1, pos2, gamma1, gamma2, suter_method=False):
    """Returns the magnetic dipolar matrix for two spins.
    Notes:
        - Position must be in cartesian coordinates.
        - Returns the frequency (not angular frequency). Nevertheless Suter considers this without division by 2pi
        as frequency and later multiplies it by a factor of 2pi. I think this is not correct.
    """
    r_vec = np.array(pos1) - np.array(pos2)
    r = np.linalg.norm(r_vec)
    n_vec = r_vec / r

    if suter_method:
        prefactor = -(c.h * c.mu_0) / (4 * np.pi * r**3) * gamma1 * gamma2
    else:
        prefactor = -(c.hbar * c.mu_0) / (4 * np.pi * r**3) * gamma1 * gamma2

    dipolar_matrix = np.zeros((3, 3))
    for i, j in product(range(3), repeat=2):

        S1_dot_n = n_vec[i]
        S2_dot_n = n_vec[j]
        S1_dot_S2 = int(i == j)

        dipolar_matrix[i, j] = prefactor * (3 * S1_dot_n * S2_dot_n - S1_dot_S2)
    return dipolar_matrix / (2 * np.pi)  # in 1/s


def calc_spin_positions(Azz, Azx, gamma1, gamma2, suter_method=False):
    r"""Calculates the spin positions from the dipolar interaction constants.

    Notes
    -----
        - C13 Position with couplings from Table V. in Zhang 2020
        Azz, Azx in [[-0.152e6, 0.110e6], [-0.198e6, 0.328e6], [-0.228e6, 0.164e6], [-0.304e6, 0.247e6]]
        - The positions are given in cartesian coordinates.
        - It is assumed that the couplings are given in units of frequency (not angular frequency).
        - Since Suter does not devide by 2pi (he uses :math:`h` instead of :math:`\hbar`), his coordinates get scaled by a factor np.cbrt(2*np.pi)=1.85.
    """

    # System of equations to solve
    def equations(variables):
        if suter_method:
            prefactor = -(c.h * c.mu_0) / (4 * np.pi) * gamma1 * gamma2 / (2 * np.pi)
        else:
            prefactor = (
                -(c.hbar * c.mu_0) / (4 * np.pi) * gamma1 * gamma2 / (2 * np.pi)
            )  #
        r, theta = variables
        eq1 = (prefactor / r**3) * (3 * np.cos(theta) ** 2 - 1) - Azz
        eq2 = (prefactor / r**3) * (3 * np.cos(theta) * np.sin(theta)) - Azx
        return [eq1, eq2]

    initial_guess = [1e-10, np.pi / 2]
    r, theta = fsolve(equations, initial_guess)
    pos = spherical_to_cartesian(r, 0, theta)
    dipolar_matrix = get_dipolar_matrix(
        (0, 0, 0), pos, gamma1, gamma2, suter_method=suter_method
    )
    if DEFAULTS["verbose"]:
        print(dipolar_matrix)

    if not np.allclose([dipolar_matrix[2, 2], dipolar_matrix[2, 0]], [Azz, Azx]):
        print("Failed")

    return pos


def calc_H_int(S1, S2, dipolar_matrix):
    """Calculates the interaction Hamiltonian between two spins from the dipolar matrix."""

    H_int_list = [
        dipolar_matrix[i, j] * S1[i + 1] * S2[j + 1]
        for i, j in product(range(3), repeat=2)
    ]
    return sum(H_int_list)


# -----------------------------------------------
