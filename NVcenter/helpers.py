from itertools import product

import numpy as np
import qutip as q
import scipy.constants as c
from scipy.optimize import fsolve

from . import DEFAULTS


# ----------------- Spin Coordinates -----------------------


def cartesian_to_spherical(x, y, z, degree=False):
    """Converts cartesian to spherical coordinates."""

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if degree:
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
    return r, phi, theta


def spherical_to_cartesian(r, phi, theta, degree=False):
    """Converts spherical to cartesian coordinates."""

    if degree:
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return float(x), float(y), float(z)


# ----------------- Spin Matrices ---------------------------------


def get_spin_matrices(spin, trunc=False):
    """Returns the spin matrices.
    Notes:
       - Includes prefactor, e.g., for spin-1/2 the returned matrices are Pauli matrices multiplied by 1/2.
       - The identity matrix is included as the first element of the list.
    """

    # natural units: energies become angular frequencies
    hbar = 1

    spin_matrices = None
    # spin-1 matrices, e.g. for the full NV-center and nitrogen nuclear spin
    if spin == 1:
        Sx = hbar * q.spin_Jx(1)
        Sy = hbar * q.spin_Jy(1)
        Sz = hbar * q.spin_Jz(1)

        # truncated matrices, e.g. for the NV center reduced to a TLS
        if trunc:
            spin_matrices = (
                q.qeye(2),
                q.Qobj(Sx[1:, 1:]),
                q.Qobj(Sy[1:, 1:]),
                q.Qobj(Sz[1:, 1:]),
            )

        else:
            spin_matrices = q.qeye(3), Sx, Sy, Sz

    # spin-1/2 matrices, e.g., for the C-13 nuclear spin
    elif spin == 1 / 2:
        sx = hbar * q.spin_Jx(1 / 2)
        sy = hbar * q.spin_Jy(1 / 2)
        sz = hbar * q.spin_Jz(1 / 2)
        spin_matrices = q.qeye(2), sx, sy, sz

    return spin_matrices


def adjust_space_dim(num_spins, operator, position):
    """Helper function to adjust the Hilbert space dimension of an operator to
    the number of spins in the system."""

    operator_list = [q.qeye(2)] * num_spins
    operator_list[position] = operator
    return q.tensor(operator_list)


# ------------------------ Magnetic dipolar interaction --------------------------------


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


# ---------------------- Analysis of the density matrix -----------------------------


def calc_logarithmic_negativity(rho, dim1=2, dim2=2):
    """Calculates the logarithmic negativity for a system of two qubits (i.e., the partial transpose wrt the second qubit)."""

    rho = rho.full()
    rho_pt = rho.copy()
    for i, j in product(range(dim1), repeat=2):
        rho_pt[1 + i * dim2, 0 + j * dim2] = rho[0 + i * dim2, 1 + j * dim2]
        rho_pt[0 + i * dim2, 1 + j * dim2] = rho[1 + i * dim2, 0 + j * dim2]
        # Note: for higher dimensions not only 0 and 1 have to be swapped
    eigv = np.linalg.eig(rho_pt)[0]
    trace_norm = sum(abs(eigv))
    return float(np.log2(trace_norm))


def calc_fidelity(rho, rho_target):
    """Calculates the a simple measure of the fidelity as overlap between the quantum states."""

    return np.abs((rho_target.dag() * rho).tr() / (rho_target.dag() * rho_target).tr())
