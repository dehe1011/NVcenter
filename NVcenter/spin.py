import numpy as np
import qutip as q

from . import CONST
from .helpers import get_spin_matrices

# -------------------------------------------


class Spin:
    r"""This class constructs the Hamiltonian and initial state from a given spin configuration.
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
        self.Bz = CONST["Bz"]

        # --------------------------------------------

        # NV center electron spin (truncated to TLS)
        if self.spin_type == "NV":  # GHz level splitting
            self.spin_dim = 2

            # Zero-field splitting
            self.D_gs = CONST["D_gs"]
            # Zeeman splitting
            self.gamma = CONST["gamma_e"]
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz

            # Frozen nitrogen spin
            self.Nzz = CONST["Nzz"]  # Dominik, Fermi contact contribution
            self.A_N = CONST["A_N"]  # Suter
            self.m_N = CONST["m_N"]  # frozen spin state of the nitrogen
            renormalization = self.m_N * self.Nzz

            # Spin operators
            self.S = get_spin_matrices(
                spin=1, trunc=True
            )  # truncated to m_s = 0 and m_s = -1

            # Hamiltonian
            self.H = (
                self.D_gs * self.S[3] ** 2
                - self.lamor * self.S[3]
                + renormalization * self.S[3]
            )

            # The NV center transition is only driven by the microwave field from outside causing the Rabi oscillations
            # self.H = q.Qobj([[0,0],[0,0]])

        # --------------------------------------------

        # C13 nuclear spin
        if self.spin_type == "C13":  # kHz level splitting
            self.spin_dim = 2

            # Zeeman splitting
            self.gamma = CONST["gamma_C"]
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # 158.5 kHz

            # Spin operators
            self.S = get_spin_matrices(spin=1 / 2)

            # Hamiltonian
            self.H = -self.lamor * self.S[3]

        # mean-field renormalization due to interation with frozen nitrogen
        # dipolar_matrix = get_dipolar_matrix((0,0,0), self.pos_C13[C13_idx], self.gamma_N, self.gamma_C13)
        # renormalization = sum([ dipolar_matrix[i,2] * self.m_N * self.S_C13[i] for i in range(3)])

        # --------------------------------------------

        if self.spin_type == "P1":
            self.spin_dim = 2

            # Zeeman splitting
            self.gamma = CONST["gamma_e"]
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz

            # Lamor frequency disorder (due to Jahn Teller effect and nitrgen spin)
            self.Jahn_Teller_dict = CONST["Jahn_Teller_dict"]
            self.lamor_disorder = (
                self.kwargs["nitrogen_spin"]
                * self.Jahn_Teller_dict[self.kwargs["axis"]]
            )

            # Spin matrices
            self.S = get_spin_matrices(spin=1 / 2)

            # Hamiltonian
            self.H = -self.lamor * self.S[3] + self.lamor_disorder * self.S[3]

        # --------------------------------------------

        if self.spin_type == "NV_full":
            self.spin_dim = 3

            # Zero-field splitting
            self.D_gs = CONST["D_gs"]
            # Zeeman splitting
            self.gamma = CONST["gamma_e"]
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # -414.8 MHz

            # Nitrogen Fermi contact contribution
            self.Nzz = CONST["Nzz"]
            self.m_N = CONST["m_N"]  # frozen spin state of the nitrogen
            renormalization = self.m_N * self.Nzz

            # Spin operators
            self.S = get_spin_matrices(spin=1)

            # Hamiltonian
            self.H = (
                self.D_gs * self.S[3] ** 2
                - self.lamor * self.S[3]
                + renormalization * self.S[3]
            )

        # --------------------------------------------

        if self.spin_type == "N":  # nitrogen-14, not nitrogen-15
            self.spin_dim = 3

            # Zero-field splitting
            self.P_gs = CONST["P_gs"]
            # Zeeman splitting
            self.gamma = CONST["gamma_N"]
            self.lamor = self.gamma * self.Bz / (2 * np.pi)  # 45.5kHz

            # Spin matrices
            self.S = get_spin_matrices(spin=1)

            # Hamiltonian
            self.H = self.P_gs * self.S[3] ** 2 - self.lamor * self.S[3]

        self.init_state = q.fock_dm(self.spin_dim, self.init_spin)
