import numpy as np
import qutip as q
from scipy.spatial.transform import Rotation as R

from .oqs import Choi_to_PTM, apply_Choi, PAULIS, apply_PTM
from .decomposition import get_decomposition
from . import verbose

# --------------------------------------------------
# [1] Rossini et al. 2023, DOI: 10.1103/PhysRevLett.131.110603
# [2] Wang et al. 2013, DOI: 10.1103/PhysRevLett.111.130504
# --------------------------------------------------

def adj_Choi(choi):
    """ See Eq. (25) in Ruskai2002. """

    # TODO: is this needed at all?   
    U23 = q.Qobj([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    adj_choi = (U23 * choi * U23).conj()
    return adj_choi

def onemax(x):
    """ Returns maximum of 1. """
    return np.real(x) if np.real(x)<1 else 0.99

def rotation_to_unitary(rotation):
    """ Converts a rotation into a unitary.
    Background: each rotation of the Bloch sphere in SO(3) corresponds to a unitary in SU(2). """

    rotvec = R.from_matrix(rotation.real).as_rotvec() # angle * rotation axis
    generator = -1j / 2 * sum(r * P for r, P in zip(rotvec, PAULIS[1:]))
    return generator.expm()

def get_circuit_input(choi):
    """ Convert a extremal Choi matrix to a quantum circuit input as described in 
    Wang2013 and Rossini2023. """

    # choi = adj_Choi(choi)
    PTM = Choi_to_PTM(choi)

    # Rossini2023 Eq.(24) & Eq. (25)
    T = PTM[1:4, 1:4]
    uu, ss, vvh = np.linalg.svd(T, full_matrices=True)
    U1 = rotation_to_unitary(np.linalg.det(uu)*uu)
    U2 = rotation_to_unitary(np.linalg.det(vvh)*vvh)
    new_T = ss/(np.linalg.det(uu) * np.linalg.det(vvh))

    # Clamp to avoid numerical issues (|cos(θ)| ≤ 1)
    cos_nu = np.real(onemax(new_T[0]))
    cos_mu = np.real(onemax(new_T[1]))
    sin_nu = np.sin(np.arccos(cos_nu))
    sin_mu = np.sin(np.arccos(cos_mu))
    
    # Compute sign sigma based on sine terms and a directional component
    t = PTM[1:4, 0]
    new_t = np.linalg.inv( np.linalg.det(uu)* uu) @ t
    sigma = np.sign(np.real(sin_nu * sin_mu)) * np.sign(np.real(new_t[2]))

    nu = 0
    if sigma == 0 or sigma == 1: 
        nu = np.arccos(cos_nu)
    if sigma == -1:
        nu = 2 * np.pi - np.arccos(cos_nu)
    mu = np.arccos(cos_mu)

    return U1, U2, np.real(mu), np.real(nu)

def run_quantum_circuit(choi, rho_init, init_gate=None, end_gate=None):
    """ Run a quantum circuit that realizes the given extremal Choi matrix. """

    U1, U2, mu, nu = get_circuit_input(choi)
    alpha = (mu + nu) / 2
    beta = (mu - nu) / 2
    angle1 = np.real(beta - alpha + np.pi / 2)
    angle2 = np.real(beta + alpha - np.pi / 2)

    # Define CNOT with flipped control/target
    cnot_flip = q.Qobj([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dims=[[2, 2], [2, 2]])

    # add gates that should not be reversed
    if init_gate is not None:
        U2 = U2 * init_gate
    if end_gate is not None:
        U1 = end_gate * U1 
        
    # Extend to ancilla qubit
    U1 = q.tensor(U1, q.qeye(2))
    U2 = q.tensor(U2, q.qeye(2))
        
    rho_init_big = q.tensor(rho_init, q.fock_dm(2, 0))

    # Apply gates
    state = U2 * rho_init_big * U2.dag()
    ry1 = q.tensor(q.qeye(2), q.gates.ry(angle1))
    state = ry1 * state * ry1.dag()
    cnot = q.gates.cnot()
    state = cnot * state * cnot.dag()
    ry2 = q.tensor(q.qeye(2), q.gates.ry(angle2))
    state = ry2 * state * ry2.dag()

    # Measure ancilla
    proj_0 = q.tensor(q.qeye(2), q.ket2dm(q.basis(2, 0)))
    proj_1 = q.tensor(q.qeye(2), q.ket2dm(q.basis(2, 1)))
    measured_states, probs = q.measurement.measurement_statistics_povm(state, [proj_0, proj_1])
    # ancilla = state.ptrace(1)

    # Apply gates
    state_0 = cnot_flip * measured_states[0] * cnot_flip.dag()
    state_1 = cnot_flip * measured_states[1] * cnot_flip.dag()
    state_0 = U1 * state_0 * U1.dag()
    state_1 = U1 * state_1 * U1.dag()
    state0, state1 = state_0.ptrace(0), state_1.ptrace(0)
    p0, p1 = probs

    if verbose:
        if init_gate is not None:
            rho_init = init_gate * rho_init * init_gate.dag()
        left_side = apply_Choi(choi, rho_init)
        if end_gate is not None:
            left_side = end_gate * left_side * end_gate.dag()
        right_side = state0 * p0 + state1 * p1
        print("Circuit successful:", np.allclose(left_side.full(), right_side.full(), atol=1e-3))

    return state0, p0, state1, p1


def apply_circuits(p, choi_p1, choi_p2, choi_m1, choi_m2, rho, init_gate=None, end_gate=None):
    """ Apply the quantum circuits that realize the given extremal Choi matrices to a density matrix. """

    state01, p01, state11, p11 = run_quantum_circuit(choi_p1, rho, init_gate, end_gate)
    state02, p02, state12, p12 = run_quantum_circuit(choi_p2, rho, init_gate, end_gate)
    state03, p03, state13, p13 = run_quantum_circuit(choi_m1, rho, init_gate, end_gate)
    state04, p04, state14, p14 = run_quantum_circuit(choi_m2, rho, init_gate, end_gate)

    state_plus = 0.5 * (state01 * p01 + state11 * p11) + 0.5 * (state02 * p02 + state12 * p12)
    state_minus = 0.5 * (state03 * p03 + state13 * p13) + 0.5 * (state04 * p04 + state14 * p14)

    return (state_plus * (1+p)/(1+2*p) - state_minus * p/(1+2*p)) * (1+2*p)


def is_circuit_successful(PTM, init_rho):
    """ Check if the mitigation is successful by comparing the initial and mitigated density matrix."""

    noisy_rho = apply_PTM(q.Qobj(PTM), init_rho)
    p, choi_p1, choi_p2, choi_m1, choi_m2 = get_decomposition(PTM)
    mitigated_rho = apply_circuits(p, choi_p1, choi_p2, choi_m1, choi_m2, noisy_rho)
    return np.allclose(init_rho.full(), mitigated_rho.full(), atol=1e-5)

# --------------------------------------------------