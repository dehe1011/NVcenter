import time
from pathos.multiprocessing import ProcessingPool  # pylint: disable=import-error
from tqdm import tqdm

import numpy as np
import qutip as q
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import DEFAULTS
from .hamiltonian import Hamiltonian, adjust_space_dim
from .spin import get_spin_matrices
from .utils import calc_fidelity, calc_logarithmic_negativity, spherical_to_cartesian

# -------------------------------------------------


def get_cnot_gate(num_qubits, control, target):
    """Returns the CNOT gate for a given number of qubits, control and target qubit."""

    qc = QuantumCircuit(num_qubits)
    qc.cx(num_qubits - 1 - control, num_qubits - 1 - target)
    dims = [[2] * num_qubits, [2] * num_qubits]
    cnot_gate = Operator(qc).data
    return q.Qobj(cnot_gate, dims=dims)


def get_hada_gate(num_qubits, target):
    """Returns the Hadamard gate for a given number of qubits and target qubit."""

    qc = QuantumCircuit(num_qubits)
    qc.h(num_qubits - 1 - target)
    dims = [[2] * num_qubits, [2] * num_qubits]
    hada_gate = Operator(qc).data
    return q.Qobj(hada_gate, dims=dims)


# -------------------------------------------------


class Pulse(Hamiltonian):
    """A class to represent a pulse sequence for a Hamiltonian system.

    Notes
    -----
        - Before and after the pulses should be a free time evolution such that
        the free time list has one more entry than the pulse time list.

    Parameters
    ----------
    pulse_seq : list
        The pulse sequence.
    register_config : dict
        Configuration for the register.
    bath_config : dict
        Configuration for the bath.
    approx_level : str
        Approximation level. Must be one of ['no_bath', 'full_bath', 'gCCE0', 'gCCE1', 'gCCE2', 'gCCE3'].
    target : object
        The target state or unitary gate.
    **kwargs : dict, optional
        - dynamical_decoupling (bool): Whether to use dynamical decoupling. Default is False.
        - old_regsiter_state (object): The old state. Default is None.
        - mode (str): Mode of operation. Must be one of ['state_preparation', 'unitary_gate']. Default is 'state_preparation'.
        - instant_pulses (bool): Whether to use instantaneous pulses. Default is False.
        - rabi_frequency (float): The Rabi frequency. Default is DEFAULTS["rabi_frequency"].
        - verbose (bool): Whether to print verbose output. Default is False.

    Important Attributes
    --------------------
    num_pulses : int
        Number of pulses in the sequence.
    total_time : float
        Total time of the pulse sequence.

    Methods
    -------
    calc_pulse_matrices(t_list)
        Calculates the pulse matrices for each system and all times in t_list.
    calc_new_states_full(t_list)
        Calculates the new states for the register for each system and all times in t_list.
    calc_fidelities_full(new_states_full)
        Calculates the fidelities for each system and all times in t_list.
    """

    def __init__(self, register_config, bath_config, **kwargs):
        self.t0 = time.time()
        self._register_config = register_config
        self._bath_config = bath_config
        self._approx_level = kwargs.get("approx_level", DEFAULTS["approx_level"])

        super().__init__(self.register_config, **kwargs)

        # Keyword arguments
        self.old_register_state = kwargs.get("old_register_state", self.register_init_state)
        self.target = kwargs.get("target", self.register_identity)
        self._pulse_seq = kwargs.get("pulse_seq", DEFAULTS["pulse_seq"])
        self.dynamical_decoupling = kwargs.get(
            "dynamical_decoupling", DEFAULTS["dynamical_decoupling"]
        )
        self.instant_pulses = kwargs.get("instant_pulses", DEFAULTS["instant_pulses"])
        self.rabi_frequency = kwargs.get("rabi_frequency", DEFAULTS["rabi_frequency"])
        self.dm_offset = kwargs.get("dm_offset", DEFAULTS["dm_offset"])
        self.num_hahn_echos = kwargs.get("num_hahn_echos", DEFAULTS["num_hahn_echos"])

        # Initialize pulse sequence
        self.init_pulse_seq()

        # Initial
        self.matrices = []
    # ------------------------------------------------

    @property
    def register_config(self):  # pylint: disable=missing-function-docstring
        return self._register_config

    @register_config.setter
    def register_config(self, new_register_config):
        if new_register_config != self._register_config:
            self._register_config = new_register_config
            super().__init__(
                self._register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

    @property
    def bath_config(self):  # pylint: disable=missing-function-docstring
        return self._bath_config

    @bath_config.setter
    def bath_config(self, new_bath_config):
        if new_bath_config != self._bath_config:
            self.pulse_matrices_full = []
            self._bath_config = new_bath_config
            self.kwargs["bath_config"] = new_bath_config
            super().__init__(
                self._register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

    @property
    def approx_level(self):  # pylint: disable=missing-function-docstring
        return self._approx_level

    @approx_level.setter
    def approx_level(self, new_approx_level):
        if new_approx_level != self._approx_level:
            self.pulse_matrices_full = []
            self._approx_level = new_approx_level
            self.kwargs["approx_level"] = new_approx_level
            super().__init__(
                self.register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

    @property
    def pulse_seq(self):  # pylint: disable=missing-function-docstring
        return self._pulse_seq

    @pulse_seq.setter
    def pulse_seq(self, new_pulse_seq):
        if list(new_pulse_seq) != list(self._pulse_seq):
            self._pulse_seq = new_pulse_seq
            self.init_pulse_seq()  # reinitialize the pulse sequence

    # ------------------------------------------------

    def init_pulse_seq(self):
        self.num_pulses = (len(self.pulse_seq) - 1) // 3
        self.free_time_list = self.pulse_seq[: self.num_pulses + 1]
        if not self.instant_pulses:
            self.pulse_time_list = self.pulse_seq[
                self.num_pulses + 1 : 2 * self.num_pulses + 1
            ]
        else:
            self.alpha_list = self.pulse_seq[
                self.num_pulses + 1 : 2 * self.num_pulses + 1
            ]
        self.phi_list = self.pulse_seq[2 * self.num_pulses + 1 :]

        self.cumulative_time_list = self.calc_cumulative_time_list()
        self.total_time = self.cumulative_time_list[-1]
        self.t_list = self.get_t_list()

        self.left_time = 0
        self.pulse_matrices_full = []

    def get_t_list(self, stepsize=0.5e-6):
        """Helper function to get the time list for a pulse sequence."""

        t_cum = self.cumulative_time_list
        t_list = []
        t_list.extend(np.arange(0, t_cum[-1], stepsize))
        pulse_time_before = list(t_cum[:-1])
        pulse_time_after = [pulse_time + 1e-9 for pulse_time in t_cum[:-1]]
        t_list.extend(pulse_time_before)
        t_list.extend(pulse_time_after)
        t_list.append(t_cum[-1])

        return np.array(list(np.real(sorted(t_list))))

    def calc_cumulative_time_list(self):
        """Calculates the cumulative time (free time evolution and pulse time)."""

        if self.instant_pulses:
            num_time_steps = len(self.free_time_list)
            cumulative_time_list = [
                sum(self.free_time_list[: i + 1]) for i in range(num_time_steps)
            ]
            return np.real(cumulative_time_list)

        num_time_steps = len(self.free_time_list) + len(self.pulse_time_list)
        full_time_list = [0] * num_time_steps
        for i, _ in enumerate(self.pulse_time_list):
            full_time_list[2 * i] = self.free_time_list[i]
            full_time_list[2 * i + 1] = self.pulse_time_list[i]
        full_time_list[-1] = self.free_time_list[-1]
        cumulative_time_list = [
            sum(full_time_list[: i + 1]) for i in range(num_time_steps)
        ]
        return np.real(cumulative_time_list)

    # ------------------------------------------------

    def calc_H_rot(self, omega, phi, theta=np.pi / 2):
        """Returns a Hamiltonian that rotates the first register spin (NV center) with the Lamor
        frequency around an axis determined by spherical angles."""

        n = np.array([spherical_to_cartesian(1, phi, theta)])
        H_rot = omega * np.sum(
            n * get_spin_matrices(1 / 2)[1:]
        )  # factor 1/2 times Pauli matrices
        H_rot = adjust_space_dim(self.system_num_spins, H_rot, 0)
        return H_rot.to(data_type="CSR")

    def calc_U_rot(self, alpha, phi, theta=np.pi / 2):
        """Returns the unitary gate that rotates the first register spin (NV center) by an
        angle alpha around an axis determined by spherical angles.

        Examples
        --------
        XGate = self.calc_U_rot(np.pi, 0, theta=np.pi/2) # -1j X
        HGate = self.calc_U_rot(np.pi, 0, theta=np.pi/4) # -1j H
        """

        t = 1  # arbitrary value bacuse it cancels
        omega = alpha / t
        H_rot = self.calc_H_rot(omega, phi, theta=theta)
        return (-1j * t * H_rot).expm()

    def calc_U_time(self, eigv, eigs, t):
        """Returns the unitary gate for the time evolution given the eigenenergies and eigenstates of an Hamiltonian."""

        U_time = eigs @ np.diag(np.exp(-1j * eigv * t)) @ eigs.conj().T
        U_time = q.Qobj(
            U_time, dims=[[2] * self.system_num_spins, [2] * self.system_num_spins]
        )
        return U_time.to(data_type="CSR")

    # ---------------------------------------------------

    def calc_eigensystem(self, free_matrix):
        """Saves the eigensystem of the free Hamiltonian and of the rotation
        if the pulses are not instantaneous."""

        eigv, eigs = [], []
        free_matrix *= 2 * np.pi  # convert to angular frequency
        eigv_free, eigs_free = np.linalg.eigh(free_matrix.full())
        eigv.append(eigv_free)
        eigs.append(eigs_free)

        if not self.instant_pulses:
            rabi_frequency = 2*np.pi*self.rabi_frequency
            for phi in self.phi_list:
                rot_matrix = self.calc_H_rot(rabi_frequency, phi, theta=np.pi/2)

                eigv_rot, eigs_rot = np.linalg.eigh((free_matrix + rot_matrix).full())
                eigv.append(eigv_rot)
                eigs.append(eigs_rot)
        return eigv, eigs

    def get_reduced_pulse_seq(self, t):
        """Returns the pulse sequence for an arbitrary time."""

        t = float(t.real)

        # if the time is larger than the total time
        if t >= self.total_time:
            free_time_list = self.free_time_list
            self.left_time = t - self.total_time
            # free_time_list[-1] += self.left_time
            if not self.instant_pulses:
                return free_time_list, self.pulse_time_list, self.phi_list

            return free_time_list, self.alpha_list, self.phi_list

        # find the time steps that are finished and the left time
        indices = [
            i + 1 for i, value in enumerate(self.cumulative_time_list) if value <= t
        ]
        finished_time_steps = indices[-1] if indices else 0
        left_time = t - self.cumulative_time_list[finished_time_steps - 1]

        # if no time step is finished
        if finished_time_steps == 0:
            return [t], [], []

        # pulse sequence for continous pulses
        if not self.instant_pulses:
            finished_free_time_steps = (
                finished_time_steps // 2 + finished_time_steps % 2
            )
            finished_pulse_time_steps = finished_time_steps // 2

            phi_list = self.phi_list[:finished_pulse_time_steps]
            pulse_time_list = self.pulse_time_list[:finished_pulse_time_steps]
            free_time_list = self.free_time_list[:finished_free_time_steps]

            if left_time >= 0 and finished_time_steps % 2 == 0:
                free_time_list.append(left_time)
            if left_time >= 0 and finished_time_steps % 2 != 0:
                pulse_time_list.append(left_time)
                phi_list.append(self.phi_list[finished_pulse_time_steps])
                free_time_list.append(
                    0
                )  # because the pulse sequence has to end with a free evolution
            # if self.verbose:
            #     print(f"Free time list: {free_time_list}, Pulse time list: {pulse_time_list}, Phi list: {phi_list}")
            return free_time_list, pulse_time_list, phi_list

        # pulse sequence for instantaneous pulses
        phi_list = self.phi_list[:finished_time_steps]
        alpha_list = self.alpha_list[:finished_time_steps]
        free_time_list = self.free_time_list[:finished_time_steps]
        if left_time >= 0:
            free_time_list.append(left_time)
        # if self.verbose:
        #     print(f"Free time list: {free_time_list}, Alpha list: {alpha_list}, Phi list: {phi_list}")
        return free_time_list, alpha_list, phi_list

    def calc_pulse_matrix(self, pulse_seq, eigv, eigs):
        """Calculates the pulse matrix for a given pulse sequence and eigensystem of an Hamiltonian."""

        # t0 = time.time()

        free_time_list, alpha_list, pulse_time_list, phi_list = [], [], [], []
        if not self.instant_pulses:
            free_time_list, pulse_time_list, phi_list = pulse_seq
        else:
            free_time_list, alpha_list, phi_list = pulse_seq
        num_pulses = len(phi_list)

        # 1. free time evolution before the first pulse
        eigv_free, eigs_free = eigv[0], eigs[0]
        first_free_evo = self.calc_U_time(eigv_free, eigs_free, free_time_list[0])
        if num_pulses == 0:
            return first_free_evo

        U_list = [first_free_evo]

        # 2. loop over pulses
        for i in range(0, num_pulses):

            if self.instant_pulses:
                # rotation
                U_rot = self.calc_U_rot(alpha_list[i], phi_list[i])

                # free time evolution
                if self.dynamical_decoupling:
                    U_half_time = self.calc_U_time(
                        eigv_free, eigs_free, free_time_list[i + 1] / 2
                    )
                    XGate = self.calc_U_rot(np.pi, 0, theta=np.pi / 2)
                    U_time = U_half_time * XGate * U_half_time
                else:
                    U_time = self.calc_U_time(
                        eigv_free, eigs_free, free_time_list[i + 1]
                    )

            else:
                # rotation
                eigv_rot, eigs_rot = eigv[i + 1], eigs[i + 1]
                U_rot = self.calc_U_time(eigv_rot, eigs_rot, pulse_time_list[i])

                # free time evolution
                if self.dynamical_decoupling:
                    difference = free_time_list[i + 1] - self.free_time_list[i + 1] / 2
                    if difference >= 0:
                        U_half_time = self.calc_U_time(
                            eigv_free, eigs_free, self.free_time_list[i + 1] / 2
                        )
                        U_difference = self.calc_U_time(
                            eigv_free, eigs_free, difference
                        )
                        # TODO: since the DD takes a finite time for continous drive one must be careful
                        # XGate = self.calc_U_time(eigv_rot, eigs_rot, 1/(2*self.rabi_frequency))
                        XGate = self.calc_U_rot(np.pi, 0, theta=np.pi / 2)
                        U_time = U_half_time * XGate * U_difference
                    else:
                        U_time = self.calc_U_time(
                            eigv_free, eigs_free, free_time_list[i + 1]
                        )
                else:
                    U_time = self.calc_U_time(
                        eigv_free, eigs_free, free_time_list[i + 1]
                    )

            U_list.append(U_rot)
            U_list.append(U_time)

        # 3. free time evolution with Hahn echos after the last pulse
        if self.left_time > 0:
            if self.num_hahn_echos == 0:
                U_list.append(self.calc_U_time(eigv_free, eigs_free, self.left_time))
            else:
                hahn_time = self.left_time / (self.num_hahn_echos + 1)
                U_hahn_time = self.calc_U_time(eigv_free, eigs_free, hahn_time)
                U_hahn = U_hahn_time
                # TODO: since the DD takes a finite time for continous drive one must be careful
                # XGate = self.calc_U_time(eigv_rot, eigs_rot, 1/(2*self.rabi_frequency))
                XGate = self.calc_U_rot(np.pi, 0, theta=np.pi / 2)
                for _ in range(self.num_hahn_echos):
                    U_hahn *= XGate * U_hahn_time
                U_list.append(U_hahn)

        # t1 = time.time()

        # 4. construct pulse_matrix from list of unitary gates
        pulse_matrix = self.system_identity  # identity
        for U in U_list[::-1]:  # see eq. (14) in Dominik's paper
            pulse_matrix *= U

        # t2 = time.time()
        # if self.verbose:
        #     print(f"Time to calculate the unitary gates: {t2-t0} s.")
        #     print(f"Time to construct the pulse matrix: {t2-t1} s.")
        return pulse_matrix

    def calc_pulse_matrices(self, i, matrices, t_list):
        eigv, eigs = self.calc_eigensystem(matrices[i])

        # loop over different timesteps for the same system
        pulse_matrices = []
        for t in t_list:
            pulse_seq = self.get_reduced_pulse_seq(t)
            pulse_matrix = self.calc_pulse_matrix(pulse_seq, eigv, eigs)

            pulse_matrices.append(pulse_matrix)

        return pulse_matrices

    def calc_pulse_matrices_full(self, t_list):
        """Calculates the pulse matrices for each system at a given time t."""

        # check if pulse matrices are already calculated
        if self.pulse_matrices_full:
            if self.verbose:
                print("Pulse matrices already calculated.")
            return self.pulse_matrices_full
        
        t0 = time.time()

        # progress bar properties
        disable_tqdm = not self.verbose
        message = f"Calculating pulse matrices for {self.approx_level}"

        # set
        if isinstance(t_list, str):
            if t_list == "final":
                t_list = [self.total_time]
            if t_list == "automatic":
                t_list = self.t_list
        self.matrices = self.calc_matrices()

        # No Parallelization
        if not self.parallelization:
            for i in tqdm(range(self.num_systems), desc=message, disable=disable_tqdm):
                self.pulse_matrices_full.append(
                    self.calc_pulse_matrices(i, self.matrices, t_list)
                )

        # Parallelization
        else:
            with ProcessingPool(processes=self.num_cpu) as pool:
                self.pulse_matrices_full = list(
                    tqdm(
                        pool.map(
                            lambda i: self.calc_pulse_matrices(
                                i, self.matrices, t_list
                            ),
                            range(self.num_systems),
                        ),
                        total=self.num_systems,
                        desc=message,
                        disable=disable_tqdm,
                    )
                )

        t1 = time.time()
        if self.verbose:
            print(f"Time to calculate the pulse matrices: {t1-t0} s.")
        
        return self.pulse_matrices_full

    # ---------------------------------------------------

    def calc_new_register_states(self, i, pulse_matrices_full, old_system_states):

        pulse_matrices = pulse_matrices_full[i]

        # loop over timesteps
        new_register_states = []
        for pulse_matrix in pulse_matrices:
            
            new_system_state = pulse_matrix * old_system_states[i] * pulse_matrix.dag()

            # reduce from system to register space by tracing out
            new_register_state = q.ptrace(
                new_system_state, np.arange(self.register_num_spins)
            )

            # offset to avoid numerical errors
            shape = new_register_state.shape
            dims = new_register_state.dims
            new_register_state += q.Qobj(np.ones(shape), dims=dims) * self.dm_offset

            new_register_states.append(new_register_state)

        return new_register_states

    def calc_new_register_states_full(self, t_list):
        """Combines Pulse Matrices and Initial States. Calculates the new states of the register for all systems and timesteps."""

        t0 = time.time()

        # progress bar properties
        disable_tqdm = not self.verbose
        message = f"Calculating new states for {self.approx_level}"

        # set
        pulse_matrices_full = self.calc_pulse_matrices_full(t_list)
        old_system_states = self.calc_system_init_states(self.old_register_state)

        # No Parallelization
        if not self.parallelization:
            new_register_states_full = []
            for i in tqdm(range(self.num_systems), desc=message, disable=disable_tqdm):
                new_register_states_full.append(
                    self.calc_new_register_states(
                        i, pulse_matrices_full, old_system_states
                    )
                )

        # Parallelization
        else:
            with ProcessingPool(processes=self.num_cpu) as pool:
                new_register_states_full = list(
                    tqdm(
                        pool.map(
                            lambda i: self.calc_new_register_states(
                                i, pulse_matrices_full, old_system_states
                            ),
                            range(self.num_systems),
                        ),
                        total=self.num_systems,
                        desc=message,
                        disable=disable_tqdm,
                    )
                )

        t1 = time.time()
        if self.verbose:
            print(f"Time to calculate the new states: {t1-t0} s.")
        return new_register_states_full

    def calc_values_full(self, observable, t_list):
        """Calculates the fidelities for the new states."""

        new_register_states_full = self.calc_new_register_states_full(t_list)

        # loop over systems
        values_full = []
        for new_register_states in new_register_states_full:

            # loop over timesteps
            values = []
            for new_register_state in new_register_states:

                if observable == "fidelity":
                    value = calc_fidelity(new_register_state, self.target)
                elif observable == "log_neg":
                    value = calc_logarithmic_negativity(new_register_state)
                else:
                    value = q.expect(new_register_state, observable)

                values.append(value)
            values_full.append(values)

        return values_full
