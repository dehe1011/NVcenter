import sympy as sp

from .oqs_sympy import STANDARD, get_vec, Choi_to_PTM, Choi_to_Kraus, Choi_to_STM

# --------------------------------------------------
# [1] Hashim 2024, arXiv:2408.12064v1
# [2] Rossini et al. 2023, DOI: 10.1103/PhysRevLett.131.110603
# [3] Ruskai et al. 2002, DOI: 10.1016/S0024-3795(01)00547-X
# --------------------------------------------------

T1, T2 = sp.symbols('T_1 T_2', real=True, positive=True)
t = sp.Symbol('t', real=True, nonnegative=True)

def is_extremal(choi):

    kraus_ops = Choi_to_Kraus(choi)
    products = []
    for i in range(len(kraus_ops)):
        for j in range(len(kraus_ops)):
            prod = kraus_ops[i].H * kraus_ops[j]
            vec = get_vec(prod)
            products.append(vec)

    if not products:
        return False

    stacked = sp.Matrix.hstack(*products)
    rank = stacked.rank()
    _is_extremal = (rank == stacked.shape[1])

    if _is_extremal:
        print("Choi matrix already describes to an extremal map.")  
    return bool(_is_extremal)

def is_ancilla_required(choi):
    """ If the STM is unitary, the map describes a quantum channel of a unitary 
    operation and can be realized without an ancilla. """

    STM = Choi_to_STM(choi)
    print("If this return an identity the choi matrix describes a unitary and can be implemented without an ancilla. ")
    return STM * STM.H

# -------------------------------------------------
# 1. Construct the STM for the forward map by process tomography
# --------------------------------------------------

def lindblad_rhs(rho, H, L_ops):
    """ Calculates the right hand side of the Lindblad equation. """

    unitary = -sp.I * (H * rho - rho * H)
    dissipator = sp.Matrix.zeros(*rho.shape)
    for L in L_ops:
        dissipator += L * rho * L.H - 0.5 * (L.H * L * rho + rho * L.H * L)
    return sp.simplify(unitary + dissipator)

def construct_STM(rho, H, L_ops, t):
    """ Constructs the superoperator/ transfer matrix in standard basis. """

    dim = rho.shape[0]
    L = sp.zeros(dim**2, dim**2)

    rhs = lindblad_rhs(rho, H, L_ops)

    for i in range(dim**2):
        basis_vec = sp.zeros(dim**2, 1)
        basis_vec[i] = 1
        rho_basis = basis_vec.reshape(dim, dim)
        rho_basis = STANDARD[i]
        
        # plug matrix unit into the rhs of the Lindblad equation
        substituted_rhs = rhs.subs({rho[i, j]: rho_basis[i, j] for i in range(dim) for j in range(dim)})
        L[:, i] = get_vec(substituted_rhs)
    L = sp.simplify(L)
    return (L * t).exp()

# --------------------------------------------------
# 2. Construct the STM for the backward map
# --------------------------------------------------

def invert_STM(STM):
    return STM.inv()

# --------------------------------------------------
# 3. Decomopose into the weighted difference of CPTP maps
# --------------------------------------------------

def simplify_choi_plus1(expr):
    return expr.replace(
                lambda e: isinstance(e, sp.Max) and e.args == (0, 1.0 - sp.exp(t/T2)),
                lambda e: 0
            )
    
def simplify_choi_plus2(expr):
    return expr.replace(
            lambda e: isinstance(e, sp.Max) and e.args == (0, 1.0 - sp.exp(t/T1)),
            lambda e: 0
        )  

def simplify_choi_minus1(expr):
    return expr.replace(
                lambda e: isinstance(e, sp.Max) and e.args == (0, sp.exp(t/T2) - 1.0),
                lambda e: sp.exp(t/T2) - 1.0
            )
    
def simplify_choi_minus2(expr):
    return expr.replace(
            lambda e: isinstance(e, sp.Max) and e.args == (0, sp.exp(t/T1) - 1.0),
            lambda e: sp.exp(t/T1) - 1.0
        )

def get_CP_Choi(choi):
    eigen_data = choi.eigenvects()
    
    choi_plus = sp.zeros(*choi.shape)
    choi_minus = sp.zeros(*choi.shape)
    
    for val, _, vecs in eigen_data:
        for v in vecs:
            norm = sp.sqrt((v.H * v)[0])
            v = v / norm if norm != 0 else v
            P = v * v.H
            choi_plus += sp.Max(0, sp.re(val)) * P
            choi_minus += sp.Max(0, -sp.re(val)) * P

    choi_plus = choi_plus.applyfunc(simplify_choi_plus1)
    choi_minus = choi_minus.applyfunc(simplify_choi_minus1)
    choi_plus = choi_plus.applyfunc(simplify_choi_plus2)
    choi_minus = choi_minus.applyfunc(simplify_choi_minus2)

    choi_plus_TP = sp.simplify( Choi_to_PTM(choi_plus) )[0, 1:]
    if choi_plus_TP == sp.Matrix([[0,0,0]]):
        print("Choi plus is already TP, no D matrix is needed")
    else: 
        print("Choi plus is not TP, choose a D matrix")
        # sp.pprint(choi_plus_TP)
        
    choi_minus_TP = sp.simplify( Choi_to_PTM(choi_minus) )[0, 1:]
    if choi_minus_TP == sp.Matrix([[0,0,0]]):
        print("Choi minus is already TP, no D matrix is needed")
    else: 
        print("Choi minus is not TP, choose a D matrix")
        # sp.pprint(choi_minus_TP)

    return sp.simplify(choi_plus), sp.simplify(choi_minus)

def get_CPTP_Choi(choi_plus, choi_minus, choi_D, p):
    choi_plus = sp.simplify( (choi_plus+choi_D)/(1+p))
    choi_minus = sp.simplify( (choi_minus+choi_D)/p)
    return choi_plus, choi_minus

# --------------------------------------------------
# 4. Decomopose into the convex sum of extremal maps
# --------------------------------------------------

# --------------------------------------------------
# 5. Combine the previous steps and perform checks 
# --------------------------------------------------

# --------------------------------------------------