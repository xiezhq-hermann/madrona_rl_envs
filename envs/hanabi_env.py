from gym.spaces import Discrete, MultiBinary

from pantheonrl_extension.multiagentenv import MultiAgentEnv
from hanabi_learning_environment.rl_env import HanabiEnv
from pantheonrl_extension.vectorenv import MadronaEnv

import build.madrona_python as madrona_python
import build.madrona_hanabi_example_python as hanabi_python

import numpy as np


DEFAULT_N = 2

FULL_CONFIG = {
            "colors":
                5,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type": 1
        }

SMALL_CONFIG = {
            "colors":
                2,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type": 1
        }

VERY_SMALL_CONFIG = {
            "colors":
                1,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type": 1
        }


DEFAULT_CONFIG = VERY_SMALL_CONFIG

config_choice = {
    'very_small': VERY_SMALL_CONFIG,
    'small': SMALL_CONFIG,
    'full': FULL_CONFIG
}


class HanabiMadrona(MadronaEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, config=None):
        self.config = (config if config is not None else DEFAULT_CONFIG)
        self.hanabi_env = HanabiEnv(config=self.config)
        observation_shape = self.hanabi_env.vectorized_observation_shape()

        # sim = None
        sim = hanabi_python.HanabiSimulator(
            exec_mode = hanabi_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            colors = config["colors"],
            ranks = config["ranks"],
            players = config["players"],
            max_information_tokens = config["max_information_tokens"],
            max_life_tokens = config["max_life_tokens"],
            debug_compile = debug_compile,
        )

        super().__init__(num_envs, gpu_id, sim)
        self.observation_space = MultiBinary(observation_shape[0])
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

        self.share_observation_space = MultiBinary(observation_shape[0] + config['ranks'] * config['colors'] * 5)
        

class PantheonHanabi(MultiAgentEnv):

    def __init__(self, config=None):
        self.config = (config if config is not None else DEFAULT_CONFIG)
        self.hanabi_env = HanabiEnv(config=self.config)

        super().__init__(ego_ind=0, n_players=self.hanabi_env.players)

        observation_shape = self.hanabi_env.vectorized_observation_shape()
        self.observation_space = MultiBinary(observation_shape[0])
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

        self.share_observation_space = MultiBinary(observation_shape[0] * 2 + 1)  # TODO

    def get_mask(self):
        legal_moves = self.hanabi_env.state.legal_moves()
        mask = [False] * self.hanabi_env.game.max_moves()
        for m in legal_moves:
            mask[self.hanabi_env.game.get_move_uid(m)] = True
        return np.array(mask, dtype=bool)

    def get_full_obs(self, obs, player):
        other_obs = np.array(obs['player_observations'][not player]['vectorized'], dtype=bool)
        my_obs = np.array(obs['player_observations'][player]['vectorized'], dtype=bool)
        player_arr = np.array([player], dtype=bool)
        if player:
            share_obs = np.concatenate((other_obs, my_obs, player_arr))
        else:
            share_obs = np.concatenate((my_obs, other_obs, player_arr))
        return my_obs, share_obs, self.get_mask()

    def n_step(self, actions):
        move = self.hanabi_env.game.get_move(actions[0]).to_dict()

        obs, reward, done, info = self.hanabi_env.step(move)

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),), tuple([reward] * self.n_players), done, info

    def n_reset(self):
        obs = self.hanabi_env.reset()

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),)

    
def validate_step(states, actions, dones, nextstates, rewards, STATIC_ENV, verbose=True):
    STATIC_ENV.n_reset()
    
    # numenvs = dones.size(0)

    # states = states.cpu().numpy()
    # actions = actions.cpu().numpy()
    # dones = dones.cpu().numpy()
    # nextstates = nextstates.cpu().numpy()
    # rewards = rewards.cpu().numpy()
    
    retval = True
    
    # for i in range(numenvs):
    #     STATIC_ENV.state[0] = states[0][i][0] - BUFFER
    #     STATIC_ENV.state[1] = states[1][i][0] - BUFFER
    #     STATIC_ENV.current_time = states[0][i][-1]
    #     STATIC_ENV.ego_state = unview(states[0][i][:TIME], STATIC_ENV.current_time)
    #     STATIC_ENV.alt_state = unview(states[1][i][:TIME], STATIC_ENV.current_time)
    #     _, truenext, truerewards, truedone, _ = STATIC_ENV.n_step(actions[:,i])
    #     truenext = np.array([truenext[0][0], truenext[1][0]])
    #     truerewards = np.array([truerewards[0], truerewards[1]])
    #     # if truedone:
    #     #     print("FINISHED EPISODE")
    #     if not np.isclose(truerewards, rewards[:, i]).all():
    #         if verbose:
    #             print("start state:", states[:, i], i)
    #             print("action:", actions[:, i])
    #             print("madrona transition:", nextstates[:, i])
    #             print("numpy transition:", truenext)
    #             print(f"Rewards mismatch: numpy={truerewards}, madrona={rewards[:, i]}")
    #         retval = False
        
    #     if truedone != dones[i]:
    #         if verbose:
    #             print("start state:", states[:, i], i)
    #             print("action:", actions[:, i])
    #             print("madrona transition:", nextstates[:, i])
    #             print("numpy transition:", truenext)
    #             print(f"DONES mismatch: numpy={truedone}, madrona={dones[i] == 1}")
    #         retval = False
    #         # return False
    #         # pass
    #     if dones[i]:
    #         # print("MADRONA DONE", nextstates[i])
    #         continue

    #     if not np.all(np.abs(truenext - nextstates[:,i]) == 0):
    #         if verbose:
    #             print("start state:", states[:, i], i)
    #             print("action:", actions[:, i])
    #             print("madrona transition:", nextstates[:, i])
    #             print("numpy transition:", truenext)
    #             print("TRANSITIONS are not equal")
    #         retval = False
    #         # return False
    #         # pass
    # print("All good")
    return retval
