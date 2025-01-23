import time 
from tqdm import tqdm
import qutip as q

from . import DEFAULTS
from .spins import Spins
from .helpers import adjust_space_dim, get_dipolar_matrix, calc_H_int

# -------------------------------------------------


class Hamiltonian(Spins):
    """
    A class to calculate the Spin Hamiltonians and initial states for all given pairs of system and mean-field spins.

    Parameters
    ----------
    register_config : list
        Configuration for the register spins.
    bath_config : list
        Configuration for the bath spins.
    approx_level : str
        Approximation level for the Hamiltonian calculation.

    Attributes
    ----------
    thermal_bath : bool
        Indicates if the bath is in a thermal state.
    suter_method : bool
        Indicates if the Suter method is used.
    system_spin_ops : list
        Spin operators I, Sx, Sy, and Sz in the correct Hilbert space dimension.
    system_identity : qutip.Qobj
        Identity operator for the system spins.
    register_init_state : qutip.Qobj
        Initial state of the register.
    system_init_states : list
        System states for a given register state.
    matrices : list
        Hamiltonian matrices for the system and mean-field spins.
    """

    def __init__(self, register_config, bath_config, approx_level, **kwargs):
        """Calculates the Spin Hamiltonians and initial states for all given pairs of system and mean-field spins."""

        t0 = time.time()
        super().__init__(register_config, bath_config, approx_level)

        # keyword arguments
        self.thermal_bath = kwargs.get("thermal_bath", DEFAULTS["thermal_bath"])
        self.suter_method = kwargs.get("suter_method", DEFAULTS["suter_method"])
        self.verbose = kwargs.get("verbose", DEFAULTS["verbose"])

        # spin operators in the larger Hilbert space
        self.register_spin_ops = self.calc_spin_ops(self.register_spins)
        self.system_spin_ops = self.calc_spin_ops(self.system_spins_list[0])

        # identity operators for register and system
        self.register_identity = q.tensor([q.qeye(2) for _ in range(self.register_num_spins)])
        self.system_identity = self.system_spin_ops[0][0]

        # initial states for register and systems
        self.register_init_state = self.calc_register_init_state()
        self.system_init_states = self.calc_system_init_states(self.register_init_state)

        # Hamiltonian matrices for the systems
        t1 = time.time()
        self.matrices = self.calc_matrices()
        t2 = time.time()
        
        if self.verbose:
            print(f"Time to calculate the Hamiltonian matrices: {t2-t1}")
            print(f"Time to initialize the Hamiltonian class: {t2-t0}")

    # -------------------------------------------------

    def calc_spin_ops(self, spins):
        """Returns the spin operators I, Sx, Sy and Sz in the Hilbert sapce of the system."""

        spin_ops = []
        for i, spin in enumerate(spins):
            spin_op = [adjust_space_dim( len(spins), op, i) for op in spin.S]
            spin_ops.append(spin_op)
        return spin_ops

    def calc_register_init_state(self):
        """Calculated the initial state of the register."""
        return q.tensor([register_spin.init_state for register_spin in self.register_spins])

    def calc_bath_init_state(self):
        """Calculates the totally mixed thermal state of the bath."""
        if self.thermal_bath:
            bath_identity = q.tensor([q.qeye(2) for _ in range(self.bath_num_spins)])
            return 1/(2**self.bath_num_spins) * bath_identity
        else:
            return q.tensor([bath_spin.init_state for bath_spin in self.bath_spins])

    def calc_system_init_states(self, register_state):
        """Returns the system states for a given register state."""

        if self.approx_level == "no_bath":
            return [register_state]

        if self.approx_level == "full_bath":
            bath_init_state = self.calc_bath_init_state()
            return [q.tensor(register_state, bath_init_state)]
        
        if self.approx_level == "gCCE0":
            return [register_state]

        states = []
        for system_spins in self.system_spins_list:
            bath_spins = system_spins[self.register_num_spins:]
            bath_init_state = q.tensor([spin.init_state for spin in bath_spins])
            states.append(q.tensor(register_state, bath_init_state))
        return states
    
    # -------------------------------------------------

    def calc_matrices(self):
        """Returns the full Hamiltonian matrices for the system and mean-field spins
        for each pair of system and mean-field."""

        matrices = []
        disable_tqdm = not self.verbose
        message = f"Calculating Hamiltonians for {self.approx_level}"

        for i in tqdm(range(self.num_systems), desc=message, disable=disable_tqdm):
            H_system = self.calc_H_system(self.system_spins_list[i])
            H_mf = self.calc_H_mf(self.system_spins_list[i], self.mf_spins_list[i])
            matrices.append(H_system + H_mf)
        return matrices

    def calc_H_system(self, system_spins):
        """Returns the Hamiltonian for the system spins."""

        system_num_spins = len(system_spins)
        H = 0
        for i, spin1 in enumerate(system_spins):
            H += adjust_space_dim(system_num_spins, spin1.H, i)
            for j, spin2 in enumerate(system_spins):
                if j > i:
                    spin_op1 = self.system_spin_ops[i]
                    spin_op2 = self.system_spin_ops[j]
                    dipolar_matrix = get_dipolar_matrix(
                        spin1.spin_pos, spin2.spin_pos, spin1.gamma, spin2.gamma, suter_method=self.suter_method
                    )

                    # the NV spin is not flipped by the surrounding spins 
                    if spin1.spin_type == "NV0":
                        dipolar_matrix[0, :] = [0, 0, 0]
                        dipolar_matrix[1, :] = [0, 0, 0]
                    H += calc_H_int(spin_op1, spin_op2, dipolar_matrix)
        return H

    def calc_H_mf(self, system_spins, mf_spins):
        """Returns the Hamiltonian for the mean-field spins (in the system Hilbert space).
        Note: This only works for a spin-1/2 bath. """

        H = 0
        for i, system_spin in enumerate(system_spins):
            Sz = self.system_spin_ops[i][3]
            for mf_spin in mf_spins:
                Ez = mf_spin.init_spin - 1 / 2  # 0,1 -> -1/2, 1/2
                dipolar_matrix = get_dipolar_matrix(
                    system_spin.spin_pos,
                    mf_spin.spin_pos,
                    system_spin.gamma,
                    mf_spin.gamma,
                )
                dipolar_component = dipolar_matrix[2, 2]
                H += Ez * dipolar_component * Sz
        return H
