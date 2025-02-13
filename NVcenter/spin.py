import numpy as np
import qutip as q

from . import CONST

# -------------------------------------------


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


# -------------------------------------------


class Spin:
    """This class constructs the Hamiltonian and initial state from a given spin configuration.
    The Hamiltonian take into account zero-field splitting (ZFS), Zeeman splitting and renormalizations due to the frozen nitrogen spin.
    For the P1 centers Lamor disorders due to the Jahn-Teller effect and the nitrogen nuclear spin are taken into consideration.

    Notes:
        - The Hamiltonian is constructed in the natural units of frequencies (not angular frequencies!).
        - The initial state is a density matrix.
        - Implemented spin types: NV, C13, P1, NV_full, N
    """

    def __init__(self, spin_type, spin_pos, init_spin, kwargs):
        # constructor arguments
        self.spin_type = spin_type
        self.spin_pos = spin_pos
        self.init_spin = init_spin
        self.kwargs = kwargs

        # Magnetic field in [111] direction of the NV center
        self.Bz = self.kwargs.get("Bz", CONST["Bz"])

        # --------------------------------------------

        # TLS NV center electron spin
        if self.spin_type == "NV":

            # Zero-field splitting
            self.D_gs = self.kwargs.get("D_gs", CONST["D_gs"])
            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_e"])
            # Nitrogen contribution
            self.N_zz = self.kwargs.get("N_zz", CONST["N_zz"])  # Dominik
            self.A_N = self.kwargs.get("A_N", CONST["A_N"])  # Suter
            # Frozen spin state of the nitrogen spin
            self.m_N = self.kwargs.get("m_N", CONST["m_N"])

            # Spin operators truncated to m_s = 0 and m_s = -1
            self.spin_dim = 2
            self.S = get_spin_matrices(spin=1, trunc=True)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz
            self.renormalization = self.m_N * self.N_zz  # 1.76 MHz
            # self.renormalization = self.m_N * self.A_N # 2.16 MHz

            self.H = (
                self.D_gs * self.S[3] ** 2
                - self.lamor * self.S[3]
                + self.renormalization * self.S[3]
            )

        # --------------------------------------------

        # TLS NV center electron spin in the rotating frame
        if self.spin_type == "NV0":

            # gyromagnetic ratio
            # needed for dipolar coupling to neighboring spins
            self.gamma = self.kwargs.get("gamma", CONST["gamma_e"])

            # Spin operators truncated to m_s = 0 and m_s = -1
            self.spin_dim = 2
            self.S = get_spin_matrices(spin=1, trunc=True)

            # Hamiltonian
            self.H = q.Qobj([[0, 0], [0, 0]])

        # --------------------------------------------

        # Surface electron spin
        if self.spin_type == "e":

            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_e"])

            # Spin operators
            self.spin_dim = 2
            self.S = get_spin_matrices(spin=1 / 2)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz
            self.H = -self.lamor * self.S[3]

        # --------------------------------------------

        # C13 nuclear spin
        if self.spin_type == "C13":  # kHz level splitting

            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_C"])

            # Spin operators
            self.spin_dim = 2
            self.S = get_spin_matrices(spin=1 / 2)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # 158.5 kHz
            self.H = -self.lamor * self.S[3]

        # mean-field renormalization due to interation with frozen nitrogen
        # dipolar_matrix = get_dipolar_matrix((0,0,0), self.pos_C13[C13_idx], self.gamma_N, self.gamma_C13)
        # renormalization = sum([ dipolar_matrix[i,2] * self.m_N * self.S_C13[i] for i in range(3)])

        # --------------------------------------------

        # P1 center electron spin with Jahn-Teller effect
        if self.spin_type == "P1":

            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_e"])
            # Lamor frequency disorder (due to Jahn Teller effect and nitrogen spin)
            self.JT_dict = self.kwargs.get("JT_dict", CONST["JT_dict"])
            self.nitrogen_spin = self.kwargs["nitrogen_spin"]
            self.axis = self.kwargs["axis"]

            # Spin matrices
            self.spin_dim = 2
            self.S = get_spin_matrices(spin=1 / 2)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz
            self.lamor_disorder = self.nitrogen_spin * self.JT_dict[self.axis]
            self.H = -self.lamor * self.S[3] + self.lamor_disorder * self.S[3]

        # --------------------------------------------

        # Full NV center electron spin (without reduction to a TLS)
        if self.spin_type == "NV_full":

            # Zero-field splitting
            self.D_gs = self.kwargs.get("D_gs", CONST["D_gs"])
            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_e"])
            # Nitrogen contribution
            self.N_zz = self.kwargs.get("N_zz", CONST["N_zz"])  # Dominik
            self.A_N = self.kwargs.get("A_N", CONST["A_N"])  # Suter
            # Frozen spin state of the nitrogen spin
            self.m_N = self.kwargs.get("m_N", CONST["m_N"])

            # Spin operators
            self.spin_dim = 3
            self.S = get_spin_matrices(spin=1, trunc=False)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz
            self.renormalization = self.m_N * self.N_zz  # 1.76 MHz
            # self.renormalization = self.m_N * self.A_N # 2.16 MHz

            self.H = (
                self.D_gs * self.S[3] ** 2
                - self.lamor * self.S[3]
                + self.renormalization * self.S[3]
            )

        # --------------------------------------------

        # Nitrogen nuclear spin (N-14, not N-15)
        if self.spin_type == "N":

            # Zero-field splitting
            self.P_gs = self.kwargs.get("P_gs", CONST["P_gs"])
            # gyromagnetic ratio
            self.gamma = self.kwargs.get("gamma", CONST["gamma_N"])

            # Spin matrices
            self.spin_dim = 3
            self.S = get_spin_matrices(spin=1)

            # Hamiltonian
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # 45.5kHz
            self.H = self.P_gs * self.S[3] ** 2 - self.lamor * self.S[3]

        # --------------------------------------------

        self.init_state = q.fock_dm(self.spin_dim, self.init_spin)


# -------------------------------------------
