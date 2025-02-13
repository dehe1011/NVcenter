from itertools import product
import numpy as np

# -----------------------------------------------


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


# -----------------------------------------------
