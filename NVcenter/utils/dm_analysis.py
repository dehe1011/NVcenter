from itertools import product
import numpy as np
import qutip as q
from tqdm import tqdm

# ----------------------------------------------------------------------


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


def get_pauli_basis(num_qubits):
    paulis = [q.qeye(2), q.sigmax(), q.sigmay(), q.sigmaz()]
    basis_ops = []

    for indices in product(range(4), repeat=num_qubits):
        ops = [paulis[i] for i in indices]
        basis_ops.append(q.tensor(ops))

    return basis_ops


def estimate_expect(M, P_in, rho):
    fid = 0
    for m in tqdm(range(M)):
        povm = [q.ket2dm(state) for state in P_in.eigenstates()[1]]
        result, _ = q.measurement.measure_povm(rho, povm)
        em_in = 2 * result - 1
        fid += em_in
    return 1/M * fid
    

def estimate_fidelity(num_qubits, N, M, rho, rho_target):

    dim = 2**num_qubits
    pauli_basis = get_pauli_basis(num_qubits)
    probabilities = [q.expect(pauli, rho_target)**2/dim for pauli in pauli_basis]
    values = np.arange(4**num_qubits)
    np.random.seed(123)
    i_n_list = np.random.choice(values, size=N, p=probabilities)
    
    fid = 0
    for n in tqdm(range(N)):
        i_n = i_n_list[n]
        P_in = pauli_basis[i_n]
        
        Pr_in = q.expect(P_in, rho_target)**2/dim
        if Pr_in == 0:
            fid += 0 
        else:
            fid += 1/Pr_in * q.expect(rho_target, P_in) * estimate_expect(M, P_in, rho)
    return 1/(dim * N) * fid

# ----------------------------------------------------------------------
