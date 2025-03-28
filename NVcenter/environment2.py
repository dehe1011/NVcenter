import time
import random
from itertools import product
import multiprocessing

from tqdm import tqdm
import numpy as np
import gymnasium as gym
import qutip as q

from . import DEFAULTS
from .evolution import Evolution
from .utils import calc_fidelity

# -------------------------------------------------


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


# -------------------------------------------------


class Environment2(Evolution, gym.Env):
    def __init__(self, register_config, **kwargs):
        super().__init__(register_config, **kwargs)
        gym.Env.__init__(self)
        self.t0 = time.time()  # print("Time elapsed:", time.time() - self.t0)

        # keyword arguments
        self.bath_configs = kwargs.get("bath_configs", [[]])
        self.num_bath_configs = len(self.bath_configs)
        self.env_approx_level = kwargs.get(
            "env_approx_level", DEFAULTS["env_approx_level"]
        )

        # Machine Learning parameters
        self.max_steps = 4
        self.infidelity_threshold = 0.1

        self.count = 0
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.current_state = self.register_init_state
        self.observation = get_observation(self.current_state)
        self.done = False

        # action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-5, high=5, shape=(32,), dtype=np.float64
        )

    # -------------------------------------------------

    def step(self, action, instant_pulses=True):
        """Apply the pulse sequence to the current state of the environment and
        return the new observation and reward."""

        # reset if episode is done (either fidelity threshold is reached or max_steps is reached)
        if self.done:
            if DEFAULTS["verbose"]:
                print("episode done")
            self.reset()
        else:
            self.count += 1

        # convert action parameters to gates
        action = [(x + 1) / 2 for x in action]
        if instant_pulses:
            self.gate_props_list = [
                ("free_evo", dict(t=5e-6 * action[0])),
                (
                    "inst_rot",
                    dict(alpha=2 * np.pi * action[1], phi=2 * np.pi * action[2]),
                ),
            ]
        else:
            self.gate_props_list = [
                ("free_evo", dict(t=1e-6 * action[0])),
                ("cont_rot", dict(t=1e-6 * action[1], phi=2 * np.pi * action[2])),
            ]

        # define the initial regsiter state as the current state
        self.old_register_states = [self.current_state]

        # choose a random bath configuration
        i = random.randint(0, self.num_bath_configs - 1)
        bath_configs = [self.bath_configs[i]]

        self.current_state = self.calc_states(bath_configs)[0, 0]
        # self.fidelity = self.calc_values('fidelity', bath_configs)[0,0]
        self.current_state = q.Qobj(self.current_state, dims=self.target.dims)
        self.fidelity = calc_fidelity(self.current_state, self.target)

        self.observation = get_observation(self.current_state)

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

        gym.Env.reset(self, seed=seed, options=options)

        self.count = 0
        self.current_state = self.register_init_state
        self.observation = get_observation(self.current_state)
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.done = False

        return self.observation, self.info

    def render(self):
        return gym.Env.render(self)

    # -------------------------------------------------

    def _get_no_bath(self, quantity, t_list, old_register_states):
        """Function to compute the bath quantities without a bath."""

        self.bath_config = []
        self.approx_level = "no_bath"

        if quantity == "states":
            states = self.get_states(t_list, old_register_states)[:, :, 0]
            if self.verbose:
                self._check_state(states.flatten())
            return states

        else:
            observable = quantity
            values = self.get_values(observable, t_list, old_register_states)[:, :, 0]
            if self.verbose and observable == "fidelity":
                self._check_fidelity(values.flatten())
            return values

    def _get_full_bath(self, bath_config, quantity, t_list, old_register_states):
        """Function to compute the bath quantities with a full bath."""

        self.bath_config = bath_config
        self.approx_level = "full_bath"

        if quantity == "states":
            states = self.get_states(t_list, old_register_states)[:, :, 0]
            if self.verbose:
                self._check_state(states.flatten())
            return states

        else:
            observable = quantity
            values = self.get_values(observable, t_list, old_register_states)[:, :, 0]
            if self.verbose and observable == "fidelity":
                self._check_fidelity(values.flatten())
            return values
        
    # -------------------------------------------------

    def _to_qutip(self, num_old_register_states, num_t_list, gCCE_states):
        """Converts the states to qutip objects."""

        for init_idx, time_idx in product(
            range(num_old_register_states), range(num_t_list)
        ):
            state = gCCE_states[init_idx, time_idx]
            gCCE_states[init_idx, time_idx] = q.Qobj(state, dims=self.register_dims)

        return gCCE_states

    def _check_state(self, states):
        """Function to check if the states are normalized and hermitian."""

        for state in states:
            norm = np.sum(state.diag())
            normalized = np.isclose(norm, 1, atol=5e-2)
            if not normalized:
                print("State not normalized.", norm)
            sx = q.expect(state, q.sigmax())
            sy = q.expect(state, q.sigmay())
            sz = q.expect(state, q.sigmaz())
            if abs(sx) > (1 + 1e-3):
                print("Sigma X: ", sx)
            if abs(sy) > (1 + 1e-3):
                print("Sigma Y: ", sy)
            if abs(sz) > (1 + 1e-3):
                print("Sigma Z: ", sz)
            hermitian = np.allclose(state.full(), state.dag().full(), atol=1e-5)
            if not hermitian:
                print("State not hermitian.")
        print("State check done.")

    def _check_fidelity(self, fidelities):
        """Function to check if the fidelities are between 0 and 1."""

        for fidelity in fidelities:
            if not 0 - 1e-1 <= fidelity <= 1 + 1e-1:
                print("Fidelity not between 0 and 1.", fidelity)
        print("Fidelity check done.")

    # -------------------------------------------------

    def _get_gCCE_states(self, bath_config, t_list, old_register_states):
        """Function to compute the bath quantities with different gCCE orders."""

        self.bath_config = bath_config

        # Initialize the gCCE states
        num_old_register_states = len(old_register_states)
        num_t_list = len(t_list)
        gCCE_states = np.zeros((num_old_register_states, num_t_list), dtype=object)

        # gCCE0
        self.approx_level = "gCCE0"
        states_gCCE0 = self.get_states(t_list, old_register_states)

        for init_idx, time_idx in product(
            range(num_old_register_states), range(num_t_list)
        ):
            # calculation of gCCE0
            gCCE_states[init_idx, time_idx] = states_gCCE0[init_idx, time_idx, 0].full()

        # return gCCE0
        if self.env_approx_level == "gCCE0":
            states = self._to_qutip(num_old_register_states, num_t_list, gCCE_states)
            if self.verbose:
                self._check_state(states.flatten())
            return states

        # gCCE1
        self.approx_level = "gCCE1"
        states_gCCE1 = self.get_states(t_list, old_register_states)

        for init_idx, time_idx in product(
            range(num_old_register_states), range(num_t_list)
        ):
            # calculation of gCCE1
            for i in self.idx_gCCE1:
                gCCE_states[init_idx, time_idx] *= (
                    states_gCCE1[init_idx, time_idx, i].full()
                    / states_gCCE0[init_idx, time_idx, 0].full()
                )

        # return gCCE1
        if self.env_approx_level == "gCCE1":
            states = self._to_qutip(num_old_register_states, num_t_list, gCCE_states)
            if self.verbose:
                self._check_state(states.flatten())
            return states

        # gCCE2
        self.approx_level = "gCCE2"
        states_gCCE2 = self.get_states(t_list, old_register_states)

        for init_idx, time_idx in product(
            range(num_old_register_states), range(num_t_list)
        ):
            # calculation of gCCE2
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                gCCE_states[init_idx, time_idx] *= (
                    states_gCCE2[init_idx, time_idx, i].full()
                    * states_gCCE0[init_idx, time_idx, 0].full()
                ) / (
                    states_gCCE1[init_idx, time_idx, j].full()
                    * states_gCCE1[init_idx, time_idx, k].full()
                )

        # return gCCE2
        if self.env_approx_level == "gCCE2":
            states = self._to_qutip(num_old_register_states, num_t_list, gCCE_states)
            if self.verbose:
                self._check_state(states.flatten())
            return states

    # -------------------------------------------------

    def get_gCCE_values(self, bath_config, observable, t_list, old_register_states):

        self.bath_config = bath_config

        # Initialize the gCCE values
        num_old_register_states = len(old_register_states)
        num_t_list = len(t_list)
        gCCE_values = np.zeros((num_old_register_states, num_t_list))

        # gCCE0
        self.approx_level = "gCCE0"
        values_gCCE0 = self.get_values(observable, t_list, old_register_states)

        # calculation of gCCE0
        gCCE_values = values_gCCE0[:, :, 0]

        # return gCCE0
        if self.env_approx_level == "gCCE0":
            if self.verbose and observable == "fidelity":
                self._check_fidelity(gCCE_values.flatten())
            return gCCE_values

        # gCCE1
        self.approx_level = "gCCE1"
        values_gCCE1 = self.get_values(
            observable, old_register_states=old_register_states, t_list=t_list
        )

        # calculation of gCCE1
        for i in self.idx_gCCE1:
            gCCE_values[:, :] *= values_gCCE1[:, :, i] / values_gCCE0[:, :, 0]

        # return gCCE1
        if self.env_approx_level == "gCCE1":
            if self.verbose and observable == "fidelity":
                self._check_fidelity(gCCE_values.flatten())
            return gCCE_values

        # gCCE2
        self.approx_level = "gCCE2"
        values_gCCE2 = self.get_values(
            observable, old_register_states=old_register_states, t_list=t_list
        )

        # calculation of gCCE2
        for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
            j, k = idx_gCCE2
            gCCE_values[:, :] *= (values_gCCE2[:, :, i] * values_gCCE0[:, :, 0]) / (
                values_gCCE1[:, :, j] * values_gCCE1[:, :, k]
            )

        # return gCCE2
        if self.env_approx_level == "gCCE2":
            if self.verbose and observable == "fidelity":
                self._check_fidelity(gCCE_values.flatten())
            return gCCE_values

    # -------------------------------------------------

    def _calc_quantity(self, bath_config, quantity, t_list, old_register_states):
        """Function to compute bath quantities in parallel."""

        if bath_config == []:
            return self._get_no_bath(quantity, t_list, old_register_states)

        if self.env_approx_level == "full_bath":
            return self._get_full_bath(bath_config, quantity, t_list, old_register_states)

        if self.env_approx_level in ["gCCE0", "gCCE1", "gCCE2"]:
            if quantity == "states":
                return self._get_gCCE_states(bath_config, t_list, old_register_states)
            else:
                observable = quantity
                return self.get_gCCE_values(bath_config, observable, t_list, old_register_states)

        return 0  # Default return if no computation was done

    def _calc_quantity_wrapper(self, args):
        """Wrapper function for multiprocessing"""
        bath_config, quantity, t_list, old_register_states = args
        return self._calc_quantity(bath_config, quantity, t_list, old_register_states)

    def calc_quantites(self, quantity, bath_configs, t_list, old_register_states):

        if bath_configs is None:
            bath_configs = self.bath_configs
        if t_list is None:
            t_list = self.t_list
        if old_register_states is None:
            old_register_states = self.old_register_states

        if self.env_approx_level == "no_bath":
            return self._get_no_bath(quantity, t_list, old_register_states)

        disable_tqdm = not True
        message = f"Sampling over spin baths..."

        quantites_baths = np.zeros(
            (len(old_register_states), len(t_list)), dtype=object
        )

        if not self.parallelization:
            for i in tqdm(
                range(len(bath_configs)), desc=message, disable=disable_tqdm
            ):
                quantites_baths += self._calc_quantity(
                    bath_configs[i], quantity, t_list, old_register_states
                )

        else:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                args = [
                    (bath_configs[i], quantity, t_list, old_register_states)
                    for i in range(len(bath_configs))
                ]
                results = list(
                    tqdm(
                        pool.imap(self._calc_quantity_wrapper, args),
                        total=self.num_bath_configs,
                        desc=message,
                        disable=disable_tqdm,
                    )
                )
            quantites_baths = sum(results)

        return 1 / len(bath_configs) * quantites_baths

    def calc_states(self, t_list=None, old_register_states=None, bath_configs=None):
        return self.calc_quantites("states", bath_configs, t_list, old_register_states)

    def calc_values(self, observable, t_list=None, old_register_states=None, bath_configs=None):
        return self.calc_quantites(observable, bath_configs, t_list, old_register_states)


# -------------------------------------------------
