import numpy as np
import qutip as q

from . import DEFAULTS
from .hamiltonian import Hamiltonian
from .helpers import calc_fidelity, spherical_to_cartesian, get_spin_matrices, adjust_space_dim

class Pulse(Hamiltonian):
    def __init__(self, pulse_seq, register_config, bath_config, approx_level, target, 
                 dynamical_decoupling=False, old_state=None, mode='state_preparation', instant_pulses=False):
        """
        Notes:
            - Note: the before and after the pulses should be a free time evolution such that 
        the free time list has one more entry than the pulse time list.
            - Available modes: 'state_preparation', 'unitary_gate'  
        """
        
        super().__init__(register_config, bath_config, approx_level)
        self.pulse_seq = pulse_seq
        self.target = target
        self.dynamical_decoupling = dynamical_decoupling
        self.mode = mode
        self.instant_pulses = instant_pulses

        # starting state (can be a quantum state or a quantum gate)
        if old_state is None and self.mode == 'state_preparation':
            self.old_states = self.calc_system_states(self.register_init_state)

        elif old_state is None and self.mode == 'unitary_gate':
            self.old_states = self.calc_system_states( q.tensor([q.qeye(2) for _ in range(self.register_num_spins)]) )

        else:
            self.old_states = self.calc_system_states(old_state)

        # Gates
        self.omega_L = DEFAULTS['omega_L']
        self.XGate = self.calc_U_rot(np.pi, 0, theta=np.pi/2) # -1j X
        self.HGate = self.calc_U_rot(np.pi, 0, theta=np.pi/4) # -1j H

        # Pulse sequence
        self.num_pulses = (len(self.pulse_seq)-1)//3
        self.free_time_list = self.pulse_seq[:self.num_pulses+1]
        if not self.instant_pulses:
            self.pulse_time_list = self.pulse_seq[self.num_pulses+1:2*self.num_pulses+1]
        else: 
            self.alpha_list = self.pulse_seq[self.num_pulses+1:2*self.num_pulses+1]
        self.phi_list = self.pulse_seq[2*self.num_pulses+1:]

        self.cumulative_time_list = self.calc_cumulative_time_list()
        self.total_time = self.cumulative_time_list[-1]

        self.pulse_matrices = self.calc_pulse_matrices(self.total_time)
        self.new_states = self.calc_new_states(self.total_time)
        self.fidelities = [calc_fidelity(dm, self.target) for dm in self.new_states]

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

    # ------------------------------------------------

    def calc_H_rot(self, omega_L, phi, theta=np.pi/2):
        """ Returns a Hamiltonian that rotates the first register spin (NV center) with the Lamor 
        frequency around an axis determined by spherical angles. """
        
        n = np.array([spherical_to_cartesian(1, phi, theta)])
        H_rot = omega_L * np.sum( n * get_spin_matrices(1/2)[1:] ) # factor 2 to get Pauli matrices
        H_rot = adjust_space_dim(self.system_num_spins, H_rot, 0)  
        return H_rot.to(data_type="CSR")

    def calc_U_rot(self, alpha, phi, theta=np.pi/2):
        """ Returns the unitary gate that rotates the first register spin (NV center) by an 
        angle alpha around an axis determined by spherical angles. """
        
        t = 1 # arbitrary value bacuse it cancels
        omega_L = alpha / t
        H_rot = self.calc_H_rot(omega_L, phi, theta=theta)
        return (-1j * t * H_rot).expm()

    def calc_U_time(self, eigv, eigs, time):
        """ Returns the unitary gate for the time evolution given the eigenenergies and eigenstates of an Hamiltonian. """
        
        U_time = eigs @ np.diag(np.exp(-1j * eigv * time)) @ eigs.conj().T
        U_time = q.Qobj(U_time, dims=[[2]*self.system_num_spins, [2] * self.system_num_spins])
        return U_time.to(data_type="CSR")

    # -------------------------------------------

    def get_reduced_pulse_seq(self, t):
        """ Returns the pulse sequence for an arbitrary time. """
        
        if t >= self.total_time:
            free_time_list = self.free_time_list
            free_time_list[-1] += t-self.total_time
            if not self.instant_pulses:
                return free_time_list, self.pulse_time_list, self.phi_list
            else: 
                return free_time_list, self.alpha_list, self.phi_list
            
        indices = [i+1 for i, value in enumerate(self.cumulative_time_list) if value <= t]
        finished_time_steps = indices[-1] if indices else 0  
        
        if finished_time_steps == 0:
            left_time = t
            return [t], [], []
            
        left_time = t - self.cumulative_time_list[finished_time_steps-1]

        finished_free_time_steps = finished_time_steps//2 + finished_time_steps%2
        finished_pulse_time_steps = finished_time_steps//2
        
        phi_list = self.phi_list[:finished_pulse_time_steps]
        pulse_time_list = self.pulse_time_list[:finished_pulse_time_steps]
        free_time_list = self.free_time_list[:finished_free_time_steps]
        
        if left_time != 0 and finished_time_steps%2==0:
            free_time_list.append(left_time)
        if left_time != 0 and finished_time_steps%2!=0:
            pulse_time_list.append(left_time)
            phi_list.append( self.phi_list[finished_pulse_time_steps] )
            free_time_list.append(0) # because the pulse sequence has to end with a free evolution
            
        return free_time_list, pulse_time_list, phi_list

    # ---------------------------------------------------

    def calc_pulse_matrix(self, pulse_seq, free_matrix):
        """ Calculates the pulse matrix for a given pulse sequence and Hamiltonian. """
        
        free_matrix *= 2*np.pi # very important!!!
        omega_L = 2*np.pi*self.omega_L # Lamor frequency as angular frequency
        eigv_free, eigs_free = np.linalg.eigh( free_matrix.full())

        if not self.instant_pulses:
            free_time_list, pulse_time_list, phi_list = pulse_seq
        else:
            free_time_list, alpha_list, phi_list = pulse_seq
        
        U_list = []
        for i in range(self.num_pulses):

            # free time evolution
            if not self.dynamical_decoupling: 
                U_time = self.calc_U_time(eigv_free, eigs_free, free_time_list[i])
            else:
                U_half_time = self.calc_U_time(eigv_free, eigs_free, free_time_list[i]/2)
                U_time = U_half_time * self.XGate * U_half_time
            U_list.append(U_time)

            # rotation 
            if not self.instant_pulses:
                rot_matrix = self.calc_H_rot(omega_L, phi_list[i])
                eigv_rot, eigs_rot = np.linalg.eigh( (free_matrix + rot_matrix).full() )
                U_rot = self.calc_U_time(eigv_rot, eigs_rot, pulse_time_list[i])
            else:
                U_rot = self.calc_U_rot(alpha_list[i], phi_list[i])
            
            U_list.append(U_rot)

        # free evolution after the last pulse
        U_list.append(self.calc_U_time(eigv_free, eigs_free, free_time_list[-1]))

        # construct pulse_matrix from list of unitary gates
        pulse_matrix = self.spin_ops[0][0] # identity
        for U in U_list[::-1]: # see eq. (14) in Dominik's paper
            pulse_matrix *= U
        return pulse_matrix

    def calc_pulse_matrices(self, t):
        """ Calculates the pulse matrices for each system at a given time t. """
        
        pulse_seq = self.get_reduced_pulse_seq(t)
        pulse_matrices = []
        for matrix in self.matrices:
            pulse_matrix = self.calc_pulse_matrix(pulse_seq, matrix) 
            pulse_matrices.append( pulse_matrix )
        return pulse_matrices

    def calc_new_states(self, t):
        """ Calculates the new states for the register at a given time t. """
        
        new_states = []
        if t == self.total_time:
            pulse_matrices = self.pulse_matrices
        else:
            pulse_matrices = self.calc_pulse_matrices(t)
        
        for pulse_matrix, old_state in zip(pulse_matrices, self.old_states):
            
            if self.mode == 'state_preparation':
                new_state = pulse_matrix * old_state * pulse_matrix.dag()
            if self.mode == 'unitary_gate':
                new_state = pulse_matrix * old_state

            # reduce from system to register space by tracing out
            reduced_new_state = q.ptrace(new_state, np.arange(self.register_num_spins))
            new_states.append(reduced_new_state)
        return new_states