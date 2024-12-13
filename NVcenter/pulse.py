import time

import numpy as np
import qutip as q

from . import DEFAULTS
from .hamiltonian import Hamiltonian
from .helpers import calc_fidelity, spherical_to_cartesian, get_spin_matrices, adjust_space_dim

# -------------------------------------------------


class Pulse(Hamiltonian):
    """ A class to represent a pulse sequence for a Hamiltonian system.

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
        - old_state (object): The old state. Default is None.
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

        # Attributes with setters
        self._approx_level = kwargs.get("approx_level", DEFAULTS["approx_level"])
        self._bath_config = bath_config
        self._pulse_seq = kwargs.get("pulse_seq", DEFAULTS["pulse_seq"])
        self._old_state = kwargs.get("old_state", DEFAULTS["old_state"])

        super().__init__(register_config, self.bath_config, self.approx_level, **kwargs)

        self._target = kwargs.get("target", self.register_identity)

        # Keyword arguments
        self.dynamical_decoupling = kwargs.get("dynamical_decoupling", DEFAULTS["dynamical_decoupling"])
        self.mode = kwargs.get("mode", "state_preparation")
        self.instant_pulses = kwargs.get("instant_pulses", DEFAULTS["instant_pulses"])
        self.rabi_frequency = kwargs.get("rabi_frequency", DEFAULTS["rabi_frequency"])
        self.verbose = kwargs.get("verbose", DEFAULTS["verbose"])

        # Initialize pulse sequence
        self.init_pulse_seq()

    # ------------------------------------------------

    @property
    def old_state(self):  # pylint: disable=missing-function-docstring
        return self._old_state

    @old_state.setter
    def old_state(self, new_old_state):
        if new_old_state != self._pulse_seq:
            self._old_state = new_old_state

    @property
    def pulse_seq(self):  # pylint: disable=missing-function-docstring
        return self._pulse_seq

    @pulse_seq.setter
    def pulse_seq(self, new_pulse_seq):
        if new_pulse_seq != self._pulse_seq:
            self._pulse_seq = new_pulse_seq
            self.init_pulse_seq()

    @property
    def approx_level(self):  # pylint: disable=missing-function-docstring
        return self._approx_level

    @approx_level.setter
    def approx_level(self, new_approx_level):
        assert new_approx_level in ['no_bath', 'full_bath', 'gCCE0', 'gCCE1', 'gCCE2', 'gCCE3'], "Invalid approximation level."
        if new_approx_level != self._approx_level:
            self._approx_level = new_approx_level
            super().__init__(self.register_config, self._bath_config, self._approx_level)

    @property
    def target(self):  # pylint: disable=missing-function-docstring
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target

    @property
    def bath_config(self):  # pylint: disable=missing-function-docstring
        return self._bath_config

    @bath_config.setter
    def bath_config(self, new_bath_config):
        if new_bath_config != self._bath_config:
            self._bath_config = new_bath_config
            super().__init__(self.register_config, self._bath_config, self._approx_level)

    # ------------------------------------------------

    def init_pulse_seq(self):
        self.num_pulses = (len(self.pulse_seq)-1)//3
        self.free_time_list = self.pulse_seq[:self.num_pulses+1]
        if not self.instant_pulses:
            self.pulse_time_list = self.pulse_seq[self.num_pulses+1:2*self.num_pulses+1]
        else: 
            self.alpha_list = self.pulse_seq[self.num_pulses+1:2*self.num_pulses+1]
        self.phi_list = self.pulse_seq[2*self.num_pulses+1:]

        self.cumulative_time_list = self.calc_cumulative_time_list()
        self.total_time = self.cumulative_time_list[-1]

    def calc_cumulative_time_list(self):
        """ Calculates the cumulative time (free time evolution and pulse time).  """
        
        if self.instant_pulses:
            num_time_steps = len(self.free_time_list)
            return [sum(self.free_time_list[:i+1]) for i in range(num_time_steps)]
            
        num_time_steps = len(self.free_time_list) + len(self.pulse_time_list)
        full_time_list = [0] * num_time_steps
        for i in range(len(self.pulse_time_list)):
            full_time_list[2*i] = self.free_time_list[i]
            full_time_list[2*i+1] = self.pulse_time_list[i]
        full_time_list[-1] = self.free_time_list[-1]
        cumulative_time_list = [sum(full_time_list[:i+1]) for i in range(num_time_steps)]
        return cumulative_time_list
    
    def get_reduced_pulse_seq(self, t):
        """ Returns the pulse sequence for an arbitrary time. """
        
        t = float(t.real)

        # if the time is larger than the total time
        if t >= self.total_time:
            if self.verbose:
                print("The time is larger than the total time.")
            free_time_list = self.free_time_list
            free_time_list[-1] += t-self.total_time
            if not self.instant_pulses:
                return free_time_list, self.pulse_time_list, self.phi_list
            else: 
                return free_time_list, self.alpha_list, self.phi_list

        # find the time steps that are finished and the left time
        indices = [i+1 for i, value in enumerate(self.cumulative_time_list) if value <= t]
        finished_time_steps = indices[-1] if indices else 0  
        left_time = t - self.cumulative_time_list[finished_time_steps-1]
        if self.verbose:
            print(f"Time: {t}, Finished time steps: {finished_time_steps}, Left time: {left_time}")
        
        # if no time step is finished
        if finished_time_steps == 0:
            if self.verbose:
                print("The time is smaller than the first time step.")
            return [t], [], []

        # pulse sequence for continous pulses
        if not self.instant_pulses:
            finished_free_time_steps = finished_time_steps//2 + finished_time_steps%2
            finished_pulse_time_steps = finished_time_steps//2
        
            phi_list = self.phi_list[:finished_pulse_time_steps]
            pulse_time_list = self.pulse_time_list[:finished_pulse_time_steps]
            free_time_list = self.free_time_list[:finished_free_time_steps]

            if left_time >= 0 and finished_time_steps%2==0:
                free_time_list.append(left_time)
            if left_time >= 0 and finished_time_steps%2!=0:
                pulse_time_list.append(left_time)
                phi_list.append( self.phi_list[finished_pulse_time_steps] )
                free_time_list.append(0) # because the pulse sequence has to end with a free evolution
            if self.verbose:
                print(f"Free time list: {free_time_list}, Pulse time list: {pulse_time_list}, Phi list: {phi_list}")
            return free_time_list, pulse_time_list, phi_list

        # pulse sequence for instantaneous pulses
        else: 
            phi_list = self.phi_list[:finished_time_steps]
            alpha_list = self.alpha_list[:finished_time_steps]
            free_time_list = self.free_time_list[:finished_time_steps]
            if left_time>=0:
                free_time_list.append(left_time)
            if self.verbose:
                print(f"Free time list: {free_time_list}, Alpha list: {alpha_list}, Phi list: {phi_list}")
            return free_time_list, alpha_list, phi_list


    # ------------------------------------------------

    def calc_H_rot(self, omega, phi, theta=np.pi/2):
        """ Returns a Hamiltonian that rotates the first register spin (NV center) with the Lamor 
        frequency around an axis determined by spherical angles. """
        
        n = np.array([spherical_to_cartesian(1, phi, theta)])
        H_rot = omega * np.sum( n * get_spin_matrices(1/2)[1:] ) # factor 1/2 times Pauli matrices
        H_rot = adjust_space_dim(self.system_num_spins, H_rot, 0)  
        return H_rot.to(data_type="CSR")

    def calc_U_rot(self, alpha, phi, theta=np.pi/2):
        """ Returns the unitary gate that rotates the first register spin (NV center) by an 
        angle alpha around an axis determined by spherical angles. 
        
        Examples
        --------
        XGate = self.calc_U_rot(np.pi, 0, theta=np.pi/2) # -1j X
        HGate = self.calc_U_rot(np.pi, 0, theta=np.pi/4) # -1j H
        """
        
        t = 1 # arbitrary value bacuse it cancels
        omega = alpha / t
        H_rot = self.calc_H_rot(omega, phi, theta=theta)
        return (-1j * t * H_rot).expm()

    def calc_U_time(self, eigv, eigs, t):
        """ Returns the unitary gate for the time evolution given the eigenenergies and eigenstates of an Hamiltonian. """
        
        U_time = eigs @ np.diag(np.exp(-1j * eigv * t)) @ eigs.conj().T
        U_time = q.Qobj(U_time, dims=[[2]*self.system_num_spins, [2] * self.system_num_spins])
        return U_time.to(data_type="CSR")

    # ---------------------------------------------------

    def calc_old_states(self):
        """ Calculates the initial states for the system. """

        if self.old_state is None and self.mode == 'state_preparation':
            return self.system_init_states

        elif self.old_state is None and self.mode == 'unitary_gate':
            register_identity = 1/(2**self.register_num_spins) * q.tensor([q.qeye(2) for _ in range(self.register_num_spins)])
            return self.calc_system_states( register_identity )

        else:
            return self.calc_system_states(self.old_state)

    def save_eigensystem(self, free_matrix):
        """ Saves the eigensystem of the free Hamiltonian and of the rotation 
        if the pulses are not instantaneous. """

        start_time = time.time()

        eigv, eigs = [], []
        free_matrix *= 2*np.pi # convert to angular frequency
        eigv_free, eigs_free = np.linalg.eigh( free_matrix.full())
        eigv.append(eigv_free)
        eigs.append(eigs_free)  

        if not self.instant_pulses:
            rabi_frequency = 2*np.pi*self.rabi_frequency # Rabi frequency as angular frequency
            for phi in self.phi_list:
                rot_matrix = self.calc_H_rot(rabi_frequency, phi)

                # remove the NV center energy barrier. this step is very important  
                # free_matrix -= q.Qobj(np.diag(free_matrix.diag()), dims=free_matrix.dims) 

                eigv_rot, eigs_rot = np.linalg.eigh( (free_matrix + rot_matrix).full() )
                eigv.append(eigv_rot)
                eigs.append(eigs_rot)

        end_time = time.time()
        if self.verbose:
            print(f"Time to calculate the eigensystem: {end_time-start_time:.2f} seconds.")
        
        # not needed because this is quite fast
        # if self.approx_level == 'full_bath':
        #     filename = os.path.join(ROOT_DIR, "NVcenter", "data", "eigensystem_full_bath.npz")
        #     np.savez(filename, eigv=eigv, eigs=eigs)
        return eigv, eigs


    def calc_pulse_matrix(self, pulse_seq, eigv, eigs):
        """ Calculates the pulse matrix for a given pulse sequence and eigensystem of an Hamiltonian. """
                
        eigv_free, eigs_free = eigv[0], eigs[0]
        if not self.instant_pulses:
            free_time_list, pulse_time_list, phi_list = pulse_seq
        else:
            free_time_list, alpha_list, phi_list = pulse_seq
        num_pulses = len(phi_list)

        U_list = []
        for i in range(num_pulses):

            # free time evolution
            if not self.dynamical_decoupling: 
                U_time = self.calc_U_time(eigv_free, eigs_free, free_time_list[i])
            else:
                U_half_time = self.calc_U_time(eigv_free, eigs_free, free_time_list[i]/2)
                XGate = self.calc_U_rot(np.pi, 0, theta=np.pi/2)
                U_time = U_half_time * XGate * U_half_time
            U_list.append(U_time)

            # rotation 
            if not self.instant_pulses:
                eigv_rot, eigs_rot = eigv[i+1], eigs[i+1]
                U_rot = self.calc_U_time(eigv_rot, eigs_rot, pulse_time_list[i])
            else:
                U_rot = self.calc_U_rot(alpha_list[i], phi_list[i])
            U_list.append(U_rot)

        # free evolution after the last pulse
        U_list.append(self.calc_U_time(eigv_free, eigs_free, free_time_list[-1]))

        # construct pulse_matrix from list of unitary gates
        pulse_matrix = self.system_identity # identity
        for U in U_list[::-1]: # see eq. (14) in Dominik's paper
            pulse_matrix *= U
        return pulse_matrix

    def calc_pulse_matrices(self, t_list):
        """ Calculates the pulse matrices for each system at a given time t. """
      
        # loop over different systems
        pulse_matrices_full = []
        for matrix in self.matrices:
            eigv, eigs = self.save_eigensystem(matrix)

            # loop over different timesteps for the same system
            pulse_matrices = []
            for t in t_list:
                pulse_seq = self.get_reduced_pulse_seq(t)
                pulse_matrix = self.calc_pulse_matrix(pulse_seq, eigv, eigs)

                pulse_matrices.append( pulse_matrix )
            pulse_matrices_full.append(pulse_matrices)
        return pulse_matrices_full

    
    # ---------------------------------------------------

    def calc_new_states_full(self, t_list):
        """ Calculates the new states for the register for all times given in t_list. """
  
        pulse_matrices = self.calc_pulse_matrices(t_list)
        old_states = self.calc_old_states()
        
        # loop over different systems
        new_states_full = []
        for i, old_state in enumerate(old_states):
            pulse_matrices_time = pulse_matrices[i]
            
            # loop over different timesteps for the same system
            new_states = []
            for pulse_matrix in pulse_matrices_time:

                assert self.mode in ['state_preparation', 'unitary_gate'], "Invalid mode."

                if self.mode == 'state_preparation':
                    new_state = pulse_matrix * old_state * pulse_matrix.dag()
                else: # self.mode == 'unitary_gate'
                    new_state = pulse_matrix * old_state

                # reduce from system to register space by tracing out
                reduced_new_state = q.ptrace(new_state, np.arange(self.register_num_spins))

                # offset to avoid numerical errors
                shape = reduced_new_state.shape
                dims = reduced_new_state.dims
                reduced_new_state += q.Qobj(np.ones(shape), dims=dims) * 1e-6

                new_states.append(reduced_new_state)
            new_states_full.append(new_states)
        return new_states_full


    def calc_fidelities_full(self, new_states_full):
        """ Calculates the fidelities for the new states. """

        fidelities_full = []
        for new_states in new_states_full:
            fidelities = [calc_fidelity(new_state, self.target) for new_state in new_states]
            fidelities_full.append(fidelities)
        return fidelities_full