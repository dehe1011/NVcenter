import qutip as q
import numpy as np
from tqdm import tqdm

from . import DEFAULTS
from .pulse import Pulse
from .helpers import calc_fidelity

# ----------------------------------------------

def numpyfy(states_list_full):
    """ Helper function to convert the states from QuTiP objects to numpy arrays needed in the gCCE formula. """

    return [[state.full() for state in states] for states in states_list_full]

def get_observation(state):
    """ Helper function to get the observation of a quantum state. """

    reshaped_state = state.full().flatten()
    return np.concatenate([np.real(reshaped_state), np.imag(reshaped_state)])

def get_state(observation, dims):
    """ Helper function to get the quantum state from an observation. """

    length = len(observation)
    real_part = observation[:length//2]
    imag_part = observation[length//2:]
    return q.Qobj(real_part, dims=dims) + 1j*q.Qobj(imag_part, dims=dims)  

# ---------------------------------------------- 

class Environment(Pulse):
    def __init__(self, register_config, bath_configs, **kwargs):

        super().__init__(register_config, [], **kwargs)

        self.bath_configs = bath_configs
        self.num_bath_configs = len(bath_configs)
        self.env_approx_level = kwargs.get('env_approx_level', DEFAULTS['env_approx_level'])
        self.approx_level = self.env_approx_level
        self.kwargs = kwargs
        
        self.max_steps = 1000
        self.infidelity_threshold = 0.001
        self.dims = self.register_init_state.dims

        self.count = 0
        self.fidelity = calc_fidelity(self.old_register_state, self.target)
        self.info = {'Fidelity': self.fidelity}
        self.reward = 0
        self.observation = get_observation(self.old_register_state)
        self.done = False

        # from gymnasium import spaces
        # self.action_space = spaces.Box(low=0, high= 1, shape=(3,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float64)
    
    # ----------------------------------------------

    def step(self, action):
        """ Apply the pulse sequence to the current state of the environment and 
        return the new observation and reward. """
        
        if self.done:
            if DEFAULTS['verbose']:
                print("episode done")
            self.reset()
        else:
            self.count += 1

        if self.instant_pulses:
            pulse_seq = [ action[0], 0, 2*np.pi*action[1], 2*np.pi*action[2]] 
        else:
            pulse_seq = [ action[0], 0, action[1], 2*np.pi*action[2]] 
        
        new_state = self.get_new_states(pulse_seq)[0]
        self.fidelity = calc_fidelity(new_state, self.target)
        self.old_state = new_state

        # observation
        self.observation = get_observation(self.old_state)

        # done
        if (self.count == self.max_steps):
            if DEFAULTS['verbose']:
                print()
            self.done = True
        if 1-self.fidelity < self.infidelity_threshold:
            self.done = True

        # reward
        if self.done:
            self.reward =  -np.log(1-self.fidelity) #-0.05*self.count
        else:
            self.reward = 0

        # info
        self.info = {'Fidelity': self.fidelity}

        return (self.observation, self.reward, self.done, self.info)
    
    def reset(self):
        """ Reset the environment to the initial state before any pulses were applied. """	
        super().__init__(self.register_config, self.bath_configs[0], **self.kwargs)

        self.count = 0
        self.fidelity = calc_fidelity(self.old_state, self.target)
        self.info = {'Fidelity': self.fidelity}
        self.reward = 0
        self.observation = get_observation(self.old_state)
        self.done = False
    
    # ----------------------------------------------

    def get_observables(self, pulse_seq, t_list=None):
        pass

    
    def get_fidelities(self, pulse_seq, t_list='final'):
        """ Implements the gCCE method to calculate the fidelities of the pulse sequence. """

        # set the pulse sequence and the t_list
        self.pulse_seq = pulse_seq
        if t_list == 'final':
                t_list = [self.total_time]
        if t_list == 'automatic':
                t_list = self.t_list
        
        # no bath or full bath
        if self.env_approx_level in ['no_bath', 'full_bath']:            
            self.approx_level = self.env_approx_level
            new_states_full = self.calc_new_states_full(t_list)
            fidelities_full = self.calc_fidelities_full(new_states_full)
            return fidelities_full[0]
        
        # iterate over all bath configurations for gCCE
        fidelities_gCCE_baths = 0 

        for bath_config in self.bath_configs:
            self.bath_config = bath_config
            
            #gCCE0 initialization
            self.approx_level = 'gCCE0'
            states_full_gCCE0 = self.calc_new_states_full(t_list)
            fidelities_full_gCCE0 = self.calc_fidelities_full(states_full_gCCE0)
            
            # gCCE0 formula
            fidelities_gCCE = fidelities_full_gCCE0[0]

            # gCCE0 result
            if self.env_approx_level == 'gCCE0':
                fidelities_gCCE_baths += np.array(fidelities_gCCE)
                continue

            # gCCE1 initialization
            self.approx_level = 'gCCE1'
            states_full_gCCE1 = self.calc_new_states_full(t_list)
            fidelities_full_gCCE1 = self.calc_fidelities_full(states_full_gCCE1)

            # gCCE1 formula
            for i, _ in enumerate(self.idx_gCCE1):
                fidelities_gCCE *= np.divide(fidelities_full_gCCE1[i], fidelities_full_gCCE0[0])
            
            # gCCE1 result
            if self.env_approx_level == 'gCCE1':
                fidelities_gCCE_baths += np.array(fidelities_gCCE)
                continue

            # gCCE2 initialization  
            self.approx_level = 'gCCE2'
            states_full_gCCE2 = self.calc_new_states_full(t_list)
            fidelities_full_gCCE2 = self.calc_fidelities_full(states_full_gCCE2)

            # gCCE2 formula
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                fidelities_gCCE *= np.divide( np.multiply(fidelities_full_gCCE2[i], fidelities_full_gCCE0[0]), np.multiply(fidelities_full_gCCE1[j], fidelities_full_gCCE1[k]) )

            # gCCE2 result 
            if self.env_approx_level == 'gCCE2':
                fidelities_gCCE_baths += np.array(fidelities_gCCE)
                continue
            
        return 1/self.num_bath_configs * fidelities_gCCE_baths


    def get_new_states(self, pulse_seq, t_list='final'):
        """ Implements the gCCE method to calculate the new states of the pulse sequence. """

        # set the pulse sequence and the t_list
        self.pulse_seq = pulse_seq
        if t_list == 'final':
                t_list = [self.total_time]
        if t_list == 'automatic':
                t_list = self.t_list
        
        # no bath or full bath
        if self.env_approx_level in ['no_bath', 'full_bath']:
            self.approx_level = self.env_approx_level
            new_states_full = self.calc_new_states_full(t_list=t_list)
            return new_states_full[0]
        
        # iterate over all bath configurations for gCCE
        states_gCCE_baths = 0 
        dims = self.register_init_state.dims

        message = f'Calculating another bath configuration'
        disable_tqdm = not self.verbose

        for bath_config in tqdm(self.bath_configs, desc=message, disable=disable_tqdm):
            self.bath_config = bath_config
            
            # gCCE0 initialization
            self.approx_level = 'gCCE0'
            states_full_gCCE0 = self.calc_new_states_full(t_list=t_list)
            states_full_gCCE0 = numpyfy(states_full_gCCE0)
            states_gCCE0 = states_full_gCCE0[0]

            states_gCCE = states_gCCE0

            # gCCE0 result
            if self.env_approx_level == 'gCCE0':
                states_gCCE_baths += np.array([q.Qobj(state, dims=dims) for state in states_gCCE])
                continue

            # gCCE1 initialization
            self.approx_level = 'gCCE1'
            states_full_gCCE1 = self.calc_new_states_full(t_list=t_list)
            states_full_gCCE1 = numpyfy(states_full_gCCE1)

            # gCCE1 formula
            for i, _ in enumerate(self.idx_gCCE1):
                states_gCCE *= np.divide(states_full_gCCE1[i], states_gCCE0)
            
            # gCCE1 result
            if self.env_approx_level == 'gCCE1':
                states_gCCE_baths += np.array([q.Qobj(state, dims=dims) for state in states_gCCE])
                continue

            # gCCE2 initialization
            self.approx_level = 'gCCE2'
            states_full_gCCE2 = self.calc_new_states_full(t_list=t_list)
            states_full_gCCE2 = numpyfy(states_full_gCCE2)

            # gCCE2 formula
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                states_gCCE *= np.divide( np.multiply(states_full_gCCE2[i], states_gCCE0), np.multiply(states_full_gCCE1[j], states_full_gCCE1[k]) )
            
            # gCCE2 result
            if self.env_approx_level == 'gCCE2':
                states_gCCE_baths += np.array([q.Qobj(state, dims=dims) for state in states_gCCE])
                continue
            
        return 1/self.num_bath_configs * states_gCCE_baths


