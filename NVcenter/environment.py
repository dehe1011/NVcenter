import time
import qutip as q
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from . import DEFAULTS
from .pulse import Pulse

# ----------------------------------------------


def numpyfy(states_list_full):
    """Helper function to convert the states from QuTiP objects to numpy arrays needed in the gCCE formula."""

    return [[state.full() for state in states] for states in states_list_full]


def get_observation(state):
    """Helper function to get the observation of a quantum state."""

    reshaped_state = state.full().flatten()
    return np.concatenate([np.real(reshaped_state), np.imag(reshaped_state)])


def get_state(observation, dims):
    """Helper function to get the quantum state from an observation."""

    length = len(observation)
    real_part = observation[: length // 2]
    imag_part = observation[length // 2 :]
    return q.Qobj(real_part, dims=dims) + 1j * q.Qobj(imag_part, dims=dims)


# ----------------------------------------------


class Environment(Pulse, gym.Env):
    def __init__(self, register_config, bath_configs, **kwargs):

        super().__init__(register_config, [], **kwargs)
        gym.Env.__init__(self)

        self.bath_configs = bath_configs
        self.num_bath_configs = len(bath_configs)
        self.env_approx_level = kwargs.get(
            "env_approx_level", DEFAULTS["env_approx_level"]
        )
        self.kwargs = kwargs

        self.max_steps = 4
        self.infidelity_threshold = 0.1
        self.dims = self.register_init_state.dims

        self.count = 0
        self.action0, self.action1, self.action2 = [], [], []
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.observation = get_observation(self.register_init_state)
        self.done = False

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(32,), dtype=np.float64
        )

    # ----------------------------------------------

    def render(self):
        return gym.Env.render(self)

    def step(self, action):
        """Apply the pulse sequence to the current state of the environment and
        return the new observation and reward."""

        if self.done:
            if DEFAULTS["verbose"]:
                print("episode done")
            self.reset()
        else:
            self.count += 1

        self.action0.append(abs(action[0]))
        if self.instant_pulses:
            self.action1.append(2 * np.pi * abs(action[1]))
        else:
            self.action1.append(action[1])
        self.action2.append(2 * np.pi * abs(action[2]))

        pulse_seq = [*self.action0 + [0], *self.action1, *self.action2]
        state = self.get_new_register_states(pulse_seq)[0]
        self.observation = np.ones(32)  # get_observation(state)

        self.fidelity = self.get_values(pulse_seq, "fidelity")[0]

        # done
        if self.count == self.max_steps:
            self.done = True
        # if 1-self.fidelity < self.infidelity_threshold:
        #     self.done = True

        # reward
        if self.done:
            self.reward = -np.log(1 - self.fidelity)  # -0.05*self.count
        else:
            self.reward = 0

        # info
        self.info = {"Fidelity": self.fidelity}

        return (self.observation, self.reward, self.done, self.done, self.info)

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state before any pulses were applied. Resets everything changed by the step function."""

        super().__init__(self.register_config, [], **self.kwargs)
        gym.Env.reset(self, seed=seed)

        self.count = 0
        self.action0, self.action1, self.action2 = [], [], []
        self.fidelity = self.get_values([0], "fidelity")[0]
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.observation = np.ones(32)  # get_observation(self.register_init_state)
        self.done = False

        return self.observation, self.info

    # ----------------------------------------------

    def get_values(self, pulse_seq, observable, t_list="final"):
        """Implements the gCCE method to calculate the fidelities of the pulse sequence."""

        # set the pulse sequence
        self.pulse_seq = pulse_seq

        # no bath or full bath
        if self.env_approx_level == "no_bath":
            self.bath_config = []
            self.approx_level = self.env_approx_level
            values_full = self.calc_values_full(observable, t_list)
            return values_full[0]

        message = "Calculating another bath configuration"
        disable_tqdm = not self.verbose

        if self.env_approx_level == "full_bath":
            values_baths = 0
            self.bath_config = self.bath_configs[0]
            self.approx_level = "full_bath"
            for i in tqdm(
                range(self.num_bath_configs), desc=message, disable=disable_tqdm
            ):
                self.bath_config = self.bath_configs[i]
                values_full = self.calc_values_full(observable, t_list)
                values_baths += np.array(values_full[0])
            return 1 / self.num_bath_configs * values_baths

        # iterate over all bath configurations for gCCE
        values_gCCE_baths = 0

        for i in tqdm(range(self.num_bath_configs), desc=message, disable=disable_tqdm):
            self.bath_config = self.bath_configs[i]

            # gCCE0 initialization
            self.approx_level = "gCCE0"
            values_full_gCCE0 = self.calc_values_full(observable, t_list)

            # gCCE0 formula
            values_gCCE = values_full_gCCE0[0]

            # gCCE0 result
            if self.env_approx_level == "gCCE0":
                values_gCCE_baths += np.array(values_gCCE)
                continue

            # gCCE1 initialization
            self.approx_level = "gCCE1"
            values_full_gCCE1 = self.calc_values_full(observable, t_list)

            # gCCE1 formula
            for i, _ in enumerate(self.idx_gCCE1):
                values_gCCE *= np.divide(values_full_gCCE1[i], values_full_gCCE0[0])

            # gCCE1 result
            if self.env_approx_level == "gCCE1":
                values_gCCE_baths += np.array(values_gCCE)
                continue

            # gCCE2 initialization
            self.approx_level = "gCCE2"
            values_full_gCCE2 = self.calc_values_full(observable, t_list)

            # gCCE2 formula
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                values_gCCE *= np.divide(
                    np.multiply(values_full_gCCE2[i], values_full_gCCE0[0]),
                    np.multiply(values_full_gCCE1[j], values_full_gCCE1[k]),
                )

            # gCCE2 result
            if self.env_approx_level == "gCCE2":
                values_gCCE_baths += np.array(values_gCCE)
                continue

        return 1 / self.num_bath_configs * values_gCCE_baths

    def get_new_register_states(self, pulse_seq, t_list="final"):
        """Implements the gCCE method to calculate the new states of the pulse sequence."""

        t0 = time.time()

        # set the pulse sequence
        self.pulse_seq = pulse_seq

        # no bath or full bath
        if self.env_approx_level == "no_bath":
            self.bath_config = []
            self.approx_level = self.env_approx_level
            new_states_full = self.calc_new_register_states_full(t_list)
            return new_states_full[0]

        message = "Calculating another bath configuration"
        disable_tqdm = not self.verbose

        if self.env_approx_level == "full_bath":
            states_baths = 0
            for i in tqdm(
                range(self.num_bath_configs), desc=message, disable=disable_tqdm
            ):
                self.bath_config = self.bath_configs[i]
                self.approx_level = "full_bath"
                states_full = self.calc_new_register_states_full(t_list)
                states_baths += np.array(
                    [q.Qobj(state, dims=self.dims) for state in states_full[0]]
                )
            return 1 / self.num_bath_configs * states_baths

        # iterate over all bath configurations for gCCE
        states_gCCE_baths = 0

        for i in tqdm(range(self.num_bath_configs), desc=message, disable=disable_tqdm):
            self.bath_config = self.bath_configs[i]

            # gCCE0 initialization
            self.approx_level = "gCCE0"
            states_full_gCCE0 = self.calc_new_register_states_full(t_list)
            states_full_gCCE0 = numpyfy(states_full_gCCE0)

            # gCCE0 formula
            states_gCCE = states_full_gCCE0[0]

            # gCCE0 result
            if self.env_approx_level == "gCCE0":
                states_gCCE_baths += np.array(
                    [q.Qobj(state, dims=self.dims) for state in states_gCCE]
                )
                continue

            # gCCE1 initialization
            self.approx_level = "gCCE1"
            states_full_gCCE1 = self.calc_new_register_states_full(t_list)
            states_full_gCCE1 = numpyfy(states_full_gCCE1)

            # gCCE1 formula
            for i, _ in enumerate(self.idx_gCCE1):
                states_gCCE *= np.divide(states_full_gCCE1[i], states_full_gCCE0[0])

            # gCCE1 result
            if self.env_approx_level == "gCCE1":
                states_gCCE_baths += np.array(
                    [q.Qobj(state, dims=self.dims) for state in states_gCCE]
                )
                continue

            # gCCE2 initialization
            self.approx_level = "gCCE2"
            states_full_gCCE2 = self.calc_new_register_states_full(t_list)
            states_full_gCCE2 = numpyfy(states_full_gCCE2)

            # gCCE2 formula
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                states_gCCE *= np.divide(
                    np.multiply(states_full_gCCE2[i], states_full_gCCE0[0]),
                    np.multiply(states_full_gCCE1[j], states_full_gCCE1[k]),
                )

            # gCCE2 result
            if self.env_approx_level == "gCCE2":
                states_gCCE_baths += np.array(
                    [q.Qobj(state, dims=self.dims) for state in states_gCCE]
                )
                continue

        t1 = time.time()
        if self.verbose:
            print(f"Elapsed time: {t1-t0} s")
        return 1 / self.num_bath_configs * states_gCCE_baths
