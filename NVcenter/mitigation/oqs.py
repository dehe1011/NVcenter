import qutip as q
import numpy as np

# --------------------------------------------------
# [1] Hashim 2024, arXiv:2408.12064v1
# --------------------------------------------------

PAULIS = [q.qeye(2), q.sigmax(), q.sigmay(), q.sigmaz()]
STANDARD  = [q.Qobj([[1,0], [0,0]]), 
               q.Qobj([[0,0], [1,0]]),
               q.Qobj([[0,1], [0,0]]),
               q.Qobj([[0,0], [0,1]])
               ]

# --------------------------------------------------
# Check CP and TP conditions for Choi matrices.
# --------------------------------------------------

def is_completely_positive(choi, tol=1e-10):
    """ Check if the Choi matrix is completely positive. 
    See p.16 (top right) in Hashim2024. """ 

    choi = choi.full()
    if not np.allclose(choi, choi.conj().T, atol=tol):
        return False  # Not Hermitian ⇒ not a valid Choi matrix
    eigvals = np.linalg.eigvalsh(choi)
    return np.all(eigvals >= -tol)

def is_trace_preserving(choi, tol=1e-10):
    """ Check if the Choi matrix is trace preserving. 
    See p.16 (top right) in Hashim2024. """

    choi = choi.full()
    tr_B = choi[0::2, 0::2] + choi[1::2, 1::2]
    return np.allclose(tr_B, np.eye(2), atol=tol)

# --------------------------------------------------
# Basis change matrices between standard and pauli representations.
# Example: pauli_to_standard() * standard_to_pauli() == q.qeye(4)
# --------------------------------------------------

def pauli_to_standard(): 
    """ Convert the Pauli basis to the standard basis. 
    See Eq. (81) in Hashim2024. """

    B = np.zeros((4,4), dtype=complex)
    for i, pauli in enumerate(PAULIS):
        B[:, i] = q.operator_to_vector(pauli).trans().full()
    return q.Qobj(B)/np.sqrt(2)

def standard_to_pauli(): 
    """ Convert the standard basis to the Pauli basis. """

    B = pauli_to_standard()
    return B.dag()

# --------------------------------------------------
# Density matrices as vectors in the Hilbert-Schmidt space B(H).
# Example: vector_to_rho( rho_to_vector(q.sigmax(), basis_type='pauli'), basis_type='pauli' ) == q.sigmax()
# --------------------------------------------------

def rho_to_vector(rho, basis_type='standard'):
    """ Convert a density matrix to a vector in the Hilbert-Schmidt space B(H). 
    Qutip implementation: q.operator_to_vector(op).
    See Eq. (114) in Hashim2024. """
    
    vec = q.Qobj([rho[0,0], rho[1,0], rho[0,1], rho[1,1]])
    if basis_type == 'pauli':
        B = pauli_to_standard()
        vec = B.dag() * vec
    return vec
        
def vector_to_rho(vec, basis_type='standard'):
    """ Convert a vector in the Hilbert-Schmidt space B(H) to a density matrix.
    Qutip implementation: q.vector_to_operator(vec). """
    
    if basis_type == 'pauli':
        B = pauli_to_standard()
        vec = B * vec
    return q.Qobj(vec.full().reshape((2, 2), order='F'))

# --------------------------------------------------
# Apply the linear dynamical map $\epsilon$ to a density matrix $\rho$.
# --------------------------------------------------

def apply_Kraus(kraus_list, rho):
    """ Apply the Kraus representation of a linear dynamical map to a density matrix.
    Qutip implementation: q.kraus(kraus_list, rho).
    See Eq. (59) in Hashim2024. """

    new_rho = 0
    for kraus in kraus_list:
        new_rho += kraus * rho * kraus.dag()
    return new_rho

def apply_STM(STM, rho):
    """ Apply the STM representation of a linear dynamical map to a density matrix. 
    See Eq. (71) in Hashim2024. """

    vec = rho_to_vector(rho)
    return vector_to_rho(STM * vec)
    
def apply_PTM(PTM, rho):
    """ Apply the PTM representation of a linear dynamical map to a density matrix.
    See Eq. (71) in Hashim2024. """

    vec = rho_to_vector(rho, basis_type='pauli')
    return vector_to_rho(PTM * vec, basis_type='pauli')

def apply_Choi(choi, rho):
    """ Apply the Choi representation of a linear dynamical map to a density matrix. 
    See Eq. (101) in Hashim2024. """

    # q.ptrace(choi * q.tensor(rho.trans(), q.qeye(2)), 0)
    new_rho = rho[0,0] * choi[0:2, 0:2] + rho[1,0] * choi[2:4, 0:2] + rho[0,1] * choi[0:2, 2:4] + rho[1,1] * choi[2:4, 2:4]
    return q.Qobj(new_rho)

# --------------------------------------------------
# Conversions between the different representations
# --------------------------------------------------

def Kraus_to_STM(kraus_list):
    """ See Eq. (73) in Hashim2024. """
    STM = 0
    for kraus_op in kraus_list:
        STM += q.tensor(kraus_op.conj(), kraus_op)
    STM.dims = [[4], [4]]
    return STM

def STM_to_PTM(STM):
    """ See Eq. (80) in Hashim2024. """
    B = pauli_to_standard()
    return B.dag() * STM * B

def PTM_to_STM(PTM):
    """ See Eq. (82) in Hashim2024. """
    B = pauli_to_standard()
    return B * PTM * B.dag()

def STM_to_Choi(STM):
    """ See Eq. (99) in Hashim2024. """
    choi = 0
    for e_ij in STANDARD:
        choi += q.tensor(e_ij, apply_STM(STM, e_ij))
    choi.dims = [[4], [4]]
    return choi

def Choi_to_STM(choi):
    """ See Eq. (72) in Hashim2024. """
    STM = np.zeros(choi.shape, dtype=complex)
    for i, e_i in enumerate(STANDARD):
        for j, e_j in enumerate(STANDARD):
            STM[i,j] += (e_i.dag() * apply_Choi(choi, e_j)).tr()
    return q.Qobj(STM)

def PTM_to_Choi(PTM):
    """ See Eq. (100) or a combination of Eq. (82) and Eq. (99)  in Hashim2024. """
    STM = PTM_to_STM(PTM)
    return STM_to_Choi(STM)

def Choi_to_PTM(choi):
    """ See Eq. (75) or a combination of Eq. (72) and Eq. (80) in Hashim2024. """
    STM = Choi_to_STM(choi)
    return STM_to_PTM(STM)

def Choi_to_Kraus(choi, tol=1e-12):
    """ Convert a Choi matrix to a Kraus representation. 
    See Eq. (112) in Hashim2024.
    Background: eigenvectors of the Choi matrix are proportional to vectorized Kraus operators."""

    eigvals, eigvecs = np.linalg.eigh(choi.full()) 
    kraus_list = []

    for i in range(len(eigvals)):
        lam = eigvals[i]
        if lam < tol:
            continue
        v = eigvecs[:, i]
        K = np.sqrt(lam) * v.reshape((2, 2), order='F')
        kraus_list.append(K)

    return kraus_list

# --------------------------------------------------