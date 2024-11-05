from itertools import product
import numpy as np
import qutip as q
import scipy.constants as c

# -------------------------------------------

# ----------------- Change coordinate system -----------------------

def cartesian_to_spherical(x, y, z, degree=False):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if degree:
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
    return r, phi, theta

def spherical_to_cartesian(r, phi, theta, degree=False):
    if degree:
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return float(x), float(y), float(z)

# ----------------- Spin matrices ---------------------------------

def get_spin_matrices(spin, trunc=False):
    """
    """
    # spin matrices in natural units (energies become frequencies)
    hbar = 1
    if spin == 1:
        # spin-1 matrices for the full NV-center and nitrogen nuclear spin
        Sx = hbar * q.spin_Jx(1)
        Sy = hbar * q.spin_Jy(1)
        Sz = hbar * q.spin_Jz(1)
        if trunc:
            # truncated matrices for the NV center: reduced to a TLS formed by the m_s = 0 and m_s = -1 state, neglecting the m_s = 1 state
            return q.qeye(2), q.Qobj(Sx[1:,1:]), q.Qobj(Sy[1:,1:]), q.Qobj(Sz[1:,1:])
        else:
            return q.qeye(3), Sx, Sy, Sz
            
    if spin == 1/2:
        # spin-1/2 matrices for the C-13 spin (Pauli matrices multiplied by 1/2)
        sx = hbar * q.spin_Jx(1/2)
        sy = hbar * q.spin_Jy(1/2)
        sz = hbar * q.spin_Jz(1/2)
        return q.qeye(2), sx, sy, sz

def adjust_space_dim(num_spins, operator, position):
    operator_list = [q.qeye(2)] * num_spins
    operator_list[position] = operator
    return q.tensor(operator_list)

# ------------------------ Magnetic dipolar interaction --------------------------------

def get_dipolar_matrix(pos1, pos2, gamma1, gamma2):
    r"""
    Notes:
        Position must be in cartesian coordinates.
        I think that this is only correct if we set $\hbar=1$, otherwise the expression should be divided by $2\pi$!
    """
    r_vec = (np.array(pos1) - np.array(pos2))
    r = np.linalg.norm( r_vec ) 
    n_vec = r_vec/r
    
    prefactor = - (c.hbar * c.mu_0) / (4 * np.pi * r**3) * gamma1 * gamma2

    dipolar_matrix = np.zeros((3, 3))
    for i, j in product(range(3), repeat=2):

        S1_dot_n = n_vec[i]
        S2_dot_n = n_vec[j]
        S1_dot_S2 = int(i == j)
        
        dipolar_matrix[i, j] = prefactor * (3 * S1_dot_n * S2_dot_n - S1_dot_S2)
    return dipolar_matrix # in Hz

def calc_H_int(S1, S2, dipolar_matrix):
    H_int_list = [dipolar_matrix[i,j] * S1[i+1] * S2[j+1] for i,j in product(range(3), repeat=2)]
    return sum(H_int_list)

# ---------------------- Analysis of the density matrix -----------------------------

def calc_logarithmic_negativity(rho, dim1=2, dim2=2):
    """ Calculates the logarithmic negativity for a system of two qubits (performs the partial transpose wrt the second qubits). """
    rho = rho.full()
    rho_pt = rho.copy()
    for i, j in product(range(dim1), repeat=2):
        rho_pt[1+i*dim2, 0+j*dim2] = rho[0+i*dim2, 1+j*dim2]
        rho_pt[0+i*dim2, 1+j*dim2] = rho[1+i*dim2, 0+j*dim2]
        # Note: for higher dimensions not only 0 and 1 have to be swapped
    eigv = np.linalg.eig(rho_pt)[0]
    trace_norm = sum(abs(eigv))
    return float(np.log2(trace_norm))

def calc_fidelity(rho, rho_target):
    """ Calculates the Fidelity. """
    return np.real((rho_target.dag()*rho).tr() / (rho_target.dag()*rho_target).tr())
