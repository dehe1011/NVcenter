import sympy as sp

# --------------------------------------------------
# [1] Hashim 2024, arXiv:2408.12064v1
# --------------------------------------------------

# Pauli matrices
I = sp.Matrix([[1, 0], [0, 1]])
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]

# matrix units
E00 = sp.Matrix([[1, 0], [0, 0]])
E10 = sp.Matrix([[0, 0], [1, 0]])
E01 = sp.Matrix([[0, 1], [0, 0]])
E11 = sp.Matrix([[0, 0], [0, 1]])
STANDARD = [E00, E10, E01, E11]

def get_vec(mat):
    rows, cols = mat.shape
    return sp.Matrix(rows * cols, 1, [mat[i, j] for j in range(cols) for i in range(rows)])

def get_unvec(vec):
    dim=2
    return sp.Matrix(dim, dim, lambda i, j: vec[j * dim + i])

# --------------------------------------------------
# Basis change matrices between standard and pauli representations.
# --------------------------------------------------

def pauli_to_standard():
    B = sp.Matrix.zeros(4, 4)
    for i, pauli in enumerate(PAULIS):
        B[:, i] = get_vec(pauli)
    return B / sp.sqrt(2)

def standard_to_pauli():
    B = pauli_to_standard()
    return B.H

# --------------------------------------------------
# Density matrices as vectors in the Hilbert-Schmidt space B(H).
# --------------------------------------------------

def rho_to_vector(rho, basis_type='standard'):
    vec = get_vec(rho)
    if basis_type == 'pauli':
        B = pauli_to_standard()
        vec = B.H * vec
    return vec

def vector_to_rho(vec, basis_type='standard'):
    if basis_type == 'pauli':
        B = pauli_to_standard()
        vec = B * vec
    return get_unvec(vec)

# --------------------------------------------------
# Apply the linear dynamical map $\epsilon$ to a density matrix $\rho$.
# --------------------------------------------------

def apply_Kraus(kraus_list, rho):
    new_rho = sp.zeros(*rho.shape)
    for K in kraus_list:
        new_rho += K * rho * K.H
    return sp.simplify(new_rho)

def apply_STM(STM, rho):
    vec = rho_to_vector(rho)
    return vector_to_rho(STM * vec)

def apply_PTM(STM, rho):
    vec = rho_to_vector(rho, basis_type='pauli')
    return vector_to_rho(STM * vec, basis_type='pauli')

def apply_Choi(choi, rho):
    new_rho = (
        rho[0, 0] * choi[0:2, 0:2] +
        rho[1, 0] * choi[2:4, 0:2] +
        rho[0, 1] * choi[0:2, 2:4] +
        rho[1, 1] * choi[2:4, 2:4]
    )
    return sp.simplify(new_rho)

# --------------------------------------------------
# Conversions between the different representations
# --------------------------------------------------

def Kraus_to_STM(kraus_list):
    STM = sp.zeros(4, 4)
    for K in kraus_list:
        STM += sp.KroneckerProduct(K.conjugate(), K).doit()
    return STM

def STM_to_PTM(STM):
    B = pauli_to_standard()
    return B.H * STM * B

def PTM_to_STM(PTM):
    B = pauli_to_standard()
    return B * PTM * B.H

def STM_to_Choi(STM):
    choi = sp.zeros(4, 4)
    for e_ij in STANDARD:
        F = apply_STM(STM, e_ij)
        choi += sp.KroneckerProduct(e_ij, F).doit()
    return choi

def Choi_to_STM(choi):
    STM = sp.zeros(4, 4)
    for i, e_i in enumerate(STANDARD):
        for j, e_j in enumerate(STANDARD):
            STM[i, j] = (e_i.H * apply_Choi(choi, e_j)).trace()
    return STM

def PTM_to_Choi(PTM):
    STM = PTM_to_STM(PTM)
    return STM_to_Choi(STM)

def Choi_to_PTM(choi):
    STM = Choi_to_STM(choi)
    return STM_to_PTM(STM)

def Choi_to_Kraus(choi): 
    eigvecs, eigvals = choi.diagonalize()
    
    kraus_list = []
    for i in range(choi.shape[0]):
        lam = eigvals[i, i]
        if lam == 0:
            continue
        v = eigvecs.col(i)
        K = sp.sqrt(lam) * get_unvec(v)
        kraus_list.append(K)
    return kraus_list

# --------------------------------------------------