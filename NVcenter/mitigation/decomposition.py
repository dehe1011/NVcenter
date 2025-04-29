
import qutip as q
import numpy as np
import scipy.linalg as la

from .oqs import Kraus_to_STM, STM_to_Choi, apply_Choi, Choi_to_STM, PTM_to_Choi, apply_PTM, is_trace_preserving, Choi_to_Kraus
from . import offset, verbose
  
# --------------------------------------------------
# [1] Hashim 2024, arXiv:2408.12064v1
# [2] Rossini et al. 2023, DOI: 10.1103/PhysRevLett.131.110603
# [3] Ruskai et al. 2002, DOI: 10.1016/S0024-3795(01)00547-X
# --------------------------------------------------

def is_extremal(choi):
    """ Check if the Choi matrix is already extremal. """

    # TODO: check if this is correct
    kraus_ops = Choi_to_Kraus(choi)

    products = []
    for i in range(len(kraus_ops)):
        for j in range(len(kraus_ops)):
            prod = kraus_ops[i].conj().T @ kraus_ops[j]
            products.append(prod.reshape((4, 1), order='F'))

    if not products:
        return False

    stacked = np.column_stack(products)
    rank = np.linalg.matrix_rank(stacked, tol=1e-5)
    _is_extremal = (rank == stacked.shape[1])

    print('Is extremal?', _is_extremal) 
    return bool(_is_extremal)

def is_ancilla_required(choi):
    """ If the STM is unitary, the map describes a quantum channel of a unitary 
    operation and can be realized without an ancilla. """

    STM = Choi_to_STM(choi)
    _is_unitary = np.allclose((STM * STM.dag()).full(), np.eye(4), atol=1e-5)
    
    print("Ancilla required?", not _is_unitary)
    return not bool(_is_unitary)

# --------------------------------------------------
# 1. Construct the PTM for the forward map by process tomography
# --------------------------------------------------

def construct_PTM(input_array):
    """ Construct the PTM from the results of the process tomography as described in section VII.B in Hashim2024.
    input_array 
        1st dimension: pauli eigenstates as initial states in order zp, zm, xp, yp (see Eq. (308) in Hashim2024).
        2nd dimension: pauli observables in order X, Y, Z (see Eq. (309) in Hashim2024).
        3rd dimension: timesteps. 
    """

    # remove the first column of the input array
    input_array = input_array[:,:,1:]

    n_timesteps = input_array.shape[-1]
    PTM_list = np.zeros((4,4,n_timesteps))

    PTM_list[0,0,:] = 1 # first column is (1,0,0,0) because the map is TP
    for i in range(3):
        zp, zm, xp, yp = input_array[0,i,:], input_array[1,i,:], input_array[2,i,:], input_array[3,i,:]
        PTM_list[i+1,0,:] = 0.5 * (zp + zm) # identity
        PTM_list[i+1,1,:] = 0.5 * (2*xp - zp - zm) # sigma_x
        PTM_list[i+1,2,:] = 0.5 * (2*yp - zp - zm) # sigma_y
        PTM_list[i+1,3,:] = 0.5 * (zp - zm) # sigma_z
    return PTM_list

def calc_noisy_rho_list(input_array, init_rho, use_old_construct_PTM=False):
    """ Applies the forward map to the initial density matrix. """

    if use_old_construct_PTM:
        PTM_list = old_construct_PTM(input_array)
    else: 
        PTM_list = construct_PTM(input_array)
    timesteps = input_array.shape[-1]

    noisy_rho_list = []
    for i in range(timesteps-1):
        PTM = PTM_list[:,:,i]
        noisy_rho = apply_PTM(q.Qobj(PTM), init_rho)
        noisy_rho_list.append(noisy_rho)
    return noisy_rho_list

def old_construct_PTM(res):
    """ Requires 18 instead of 12 measurements. Kept for backward compatibility. """

    n_timesteps = res.shape[-1]
    timematrix = np.zeros((4,4,n_timesteps))
    timematrix[0,0,:] = 2     
    for i in range(3):
        timematrix[i+1,0,:] = res[4,i,:]+res[5,i,:]
        for j in range(3):
            timematrix[i+1,j+1,:] = res[2*j,i,:]-res[2*j+1,i,:] 
    return timematrix/2

# --------------------------------------------------
# 2. Construct the PTM for the backward map
# --------------------------------------------------

def invert_PTM(PTM):
    """ Construct the inverse PTM to invert the map. """

    PTM_inv = np.zeros((4, 4))
    PTM_inv[0, 0] = 1  # top-left stays 1

    t = PTM[1:, 0]
    R = PTM[1:, 1:] 
    R_inv = np.linalg.inv(R)
    t_inv = -np.dot(R_inv, t)
    
    PTM_inv[1:, 0] = t_inv
    PTM_inv[1:, 1:] = R_inv

    return PTM_inv

# --------------------------------------------------
# 3. Decomopose into the weighted difference of CPTP maps
# --------------------------------------------------

def get_CP_Choi(choi):
    """ Decompose the Choi matrix into the difference of two CP maps.
    See Eq. (15) in Rossini2023. """

    eigv, eigs = np.linalg.eig(choi.full())
    projectors = [np.outer(v, v.conj()) for v in eigs.T]
    choi_plus = sum(max(0, ev.real) * P for ev, P in zip(eigv, projectors))
    choi_minus = sum(max(0, -ev.real) * P for ev, P in zip(eigv, projectors))

    # add offset for numerical stability
    if np.sum(np.abs(choi_minus)) != 0:
        reg = offset * sum(projectors)
        choi_plus += reg
        choi_minus += reg

    return q.Qobj(choi_plus), q.Qobj(choi_minus)

def get_p(choi_minus): 
    """ Find the p value.
     See Eq. (12) in Rossini2023. """

    kraus_sum = choi_minus[0::2, 0::2] + choi_minus[1::2, 1::2]
    eigv, _ = np.linalg.eig(kraus_sum)
    return max(eigv + offset).real, kraus_sum

def get_Choi_D(choi_minus):
    """ Construct the D matrix to make the Choi matrices TP. 
    See Eq. (13) in Rossini2023. """

    p, kraus_sum = get_p(choi_minus)
    if is_trace_preserving(choi_minus/p):
        if verbose:
            print("Info: choi_minus/p is TP, so we choose the zero matrix for D.")
            print("---------------------------------------")
        return p, q.Qobj(np.zeros((4, 4)))

    # Check if kraus_sum can have off-diagonal elements (I assume this is the case)
    # If kraus_sum is diagonal, the STM_D contains the difference of the kraus_sum diagonal entries in the top left or bottom right corner. 
    if verbose: 
        is_diagonal = np.allclose(kraus_sum[0,1]+kraus_sum[1,0], 0, atol=1e-3)
        print("kraus_sum is diagonal", is_diagonal)

    D_dag_D = p * np.eye(2) - kraus_sum
    D = q.Qobj(la.sqrtm(D_dag_D))
    STM_D = Kraus_to_STM([D])
    choi_D = STM_to_Choi(STM_D)
    return p, choi_D

def get_CPTP_Choi(choi):
    choi_plus, choi_minus = get_CP_Choi(choi)
    p, choi_D = get_Choi_D(choi_minus)

    choi_plus = (choi_plus+choi_D)/(1+p)
    choi_minus = (choi_minus+choi_D)/p
        
    return p, choi_plus, choi_minus

# --------------------------------------------------
# 4. Decomopose into the convex sum of extremal maps
# --------------------------------------------------

def get_extremal_Choi(choi):
    """ Convert the Choi matrix into the convex sum of two extremal maps following Ruskai2002.
    Ruskai2002:
        Lemma 1: CPTP choi matrix describes by a contraction matrix R
        Proposition 1: For extremal maps the contraction R is unitary
        Lemma 2: How to find the convex sum of extremal maps by diving the contraction matrix. 
    """
    
    choi = choi.full()

    # determine A, B, C matrices
    A = choi[0:2, 0:2]
    B = choi[2:4, 2:4]
    C = choi[0:2, 2:4]

    # calculate contraction matrix R
    # TODO: this could be numerically unstable
    sqrtA = la.sqrtm(A)
    sqrtB = la.sqrtm(B)
    R = la.inv(sqrtA) @ C @ la.inv(sqrtB)
    
    u, s, vh = np.linalg.svd(R)
    # s = np.round(s, 5)
    theta1, theta2 = np.arccos(s)
    D = np.diag([np.exp(1j * theta1), np.exp(1j * theta2)])

    # Create Choi1
    choi1 = choi.copy()
    block1 = sqrtA @ u @ D @ vh @ sqrtB
    choi1[0:2, 2:4] = block1
    choi1[2:4, 0:2] = block1.conj().T

    # Create Choi2
    choi2 = choi.copy()
    block2 = sqrtA @ u @ D.conj() @ vh @ sqrtB
    choi2[0:2, 2:4] = block2
    choi2[2:4, 0:2] = block2.conj().T

    if verbose:
        print("Checking extremal maps...")
        if np.allclose(choi1, choi2, atol=1e-5):
            print("Choi matrix already belongs to an extremal map. It is not necessary to ditribute further into two choi matrices.")  

    return q.Qobj(choi1), q.Qobj(choi2)

# --------------------------------------------------
# 5. Combine the previous steps and perform checks 
# --------------------------------------------------
    
def get_decomposition(PTM):
    """ Constructs the decomposition of the inverse map in extremal maps that can be 
    simulated efficiently according to Wang2013. """
    
    inv_PTM = invert_PTM(PTM)
    inv_choi = PTM_to_Choi(q.Qobj(inv_PTM))

    p, choi_plus, choi_minus = get_CPTP_Choi(inv_choi)
    if verbose:
        left_side = inv_choi
        right_side = (1+p) * choi_plus - p * choi_minus
        print("Checking map divisions...")
        print("Division 1 successful: ", np.allclose(left_side.full(), right_side.full(), atol=1e-5))
        print("---------------------------------------")
    
    choi_p1, choi_p2 = get_extremal_Choi(choi_plus)
    if verbose:
        left_side = choi_plus
        right_side = 0.5 * choi_p1 + 0.5 * choi_p2
        print("Division 2_1 successful: ", np.allclose(left_side.full(), right_side.full(), atol=1e-5))
        print("---------------------------------------")
            
    choi_m1, choi_m2 = get_extremal_Choi(choi_minus)
    if verbose:
        left_side = choi_minus
        right_side = 0.5 * choi_m1 + 0.5 * choi_m2
        print("Division 2_2 successful: ", np.allclose(left_side.full(), right_side.full(), atol=1e-5))
        print("---------------------------------------")

    if verbose:
        choi_plus = 0.5 * choi_p1 + 0.5 * choi_p2
        choi_minus = 0.5 * choi_m1 + 0.5 * choi_m2
        left_side = inv_choi
        right_side = (1+p) * choi_plus - p * choi_minus
        print("Full division successful: ", np.allclose( left_side.full(), right_side.full() ))
        print("---------------------------------------")

    return p, choi_p1, choi_p2, choi_m1, choi_m2


def apply_decomposition(p, choi_p1, choi_p2, choi_m1, choi_m2, rho):
    """ Apply the decomposed Choi matrix to a density matrix."""

    choi_plus = 0.5 * choi_p1 + 0.5 * choi_p2
    choi_minus = 0.5 * choi_m1 + 0.5 * choi_m2
    choi = (1+p) * choi_plus - p * choi_minus
    return apply_Choi(choi, rho)


def is_decomposition_successful(PTM, init_rho):
    """ Check if the mitigation is successful by comparing the initial and mitigated density matrix."""

    noisy_rho = apply_PTM(q.Qobj(PTM), init_rho)
    p, choi_p1, choi_p2, choi_m1, choi_m2 = get_decomposition(PTM)
    mitigated_rho = apply_decomposition(p, choi_p1, choi_p2, choi_m1, choi_m2, noisy_rho)
    return np.allclose(init_rho.full(), mitigated_rho.full(), atol=1e-5)

# --------------------------------------------------