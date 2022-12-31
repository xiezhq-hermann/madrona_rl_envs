from gym.spaces import Discrete, MultiDiscrete
import build.madrona_python as madrona_python
import build.madrona_balance_example_python as balance_python

from .multiagentenv import MultiAgentEnv

import numpy as np

import torch

NUM_SPACES = 5
VALID_MOVES = [-2, -1, 1, 2]
BUFFER = 2
TIME = 3

SCALE = 0.2

MAX_STATES = (NUM_SPACES ** 2)

def to_torch(a):
    return a.detach().clone()

class BalanceMadronaTorch():

    def __init__(self, num_envs, gpu_id, debug_compile=True):
        self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.num_envs = num_envs
        # super().__init__(num_envs, observation_space, action_space)

        self.sim = balance_python.BalanceBeamSimulator(
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
        )

        self.static_dones = self.sim.done_tensor().to_torch()
        
        self.static_active_agents = self.sim.active_agent_tensor().to_torch()
        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_agent_states = self.sim.agent_state_tensor().to_torch()
        self.static_action_masks = self.sim.action_mask_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()
        self.static_worldID = self.sim.world_id_tensor().to_torch().to(torch.long)

        # verify all agent IDs are as expected
        self.static_agentID = self.sim.agent_id_tensor().to_torch().to(torch.long)

        # print(self.sim.world_id_tensor().to_torch())

        # self.static_gathered_dones = self.static_dones.detach().clone()

        self.static_scattered_active_agents = self.static_active_agents.detach().clone()
        self.static_scattered_observations = self.static_observations.detach().clone()
        self.static_scattered_agent_states = self.static_agent_states.detach().clone()
        self.static_scattered_action_masks = self.static_action_masks.detach().clone()
        self.static_scattered_rewards = self.static_rewards.detach().clone()

        # self.static_scattered_active_agents.scatter_(1, self.static_worldID[:,:], self.static_active_agents)
        # self.static_scattered_observations.scatter_(1, self.static_worldID[:,:,None].expand(self.static_observations.size()), self.static_observations)
        # self.static_scattered_agent_states.scatter_(1, self.static_worldID[:,:,None].expand(self.static_agent_states.size()), self.static_agent_states)
        # self.static_scattered_action_masks.scatter_(1, self.static_worldID[:,:,None].expand(self.static_action_masks.size()), self.static_action_masks)
        # self.static_scattered_rewards.scatter_(1, self.static_worldID[:,:], self.static_rewards)

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        self.infos = [{}] * self.num_envs
        # print("Dones", self.static_dones)
        # print("Obs", self.static_observations)
        # print("Corrected Obs", self.static_scattered_observations)
        # print("Rew", self.static_rewards)
        # print("WorldID", self.static_worldID)
        # print("AgentID", self.static_agentID)

    def step(self, actions):
        self.static_actions.copy_(actions[self.static_agentID, self.static_worldID, :])
        # torch.gather(actions, 1, self.static_worldID[:,:,None], out=self.static_actions)
        
        self.sim.step()

        # torch.gather(self.static_dones, 0, self.static_worldID[0], out=self.static_gathered_dones)
        # torch.gather(self.static_active_agents, 1, self.static_worldID, out=self.static_gathered_actives)

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards
        
        # self.static_scattered_active_agents.scatter_(1, self.static_worldID[:,:], self.static_active_agents)
        # self.static_scattered_observations.scatter_(1, self.static_worldID[:,:,None].expand(self.static_observations.size()), self.static_observations)
        # self.static_scattered_agent_states.scatter_(1, self.static_worldID[:,:,None].expand(self.static_agent_states.size()), self.static_agent_states)
        # self.static_scattered_action_masks.scatter_(1, self.static_worldID[:,:,None].expand(self.static_action_masks.size()), self.static_action_masks)
        # self.static_scattered_rewards.scatter_(1, self.static_worldID[:,:], self.static_rewards)

        return to_torch(self.static_scattered_observations), to_torch(self.static_scattered_rewards), to_torch(self.static_dones), self.infos
        # return to_torch(self.static_observations), to_torch(self.static_rewards), to_torch(self.static_gathered_dones), self.infos

    def reset(self):
        return to_torch(self.static_scattered_observations)

    def close(self, **kwargs):
        pass

def generate_state(index):
    # index = index // 2 + 1
    x = index % NUM_SPACES
    y = index // NUM_SPACES
    return np.array([x, y], dtype=int)
    # return np.random.randint(0, NUM_SPACES, 2)

def view(state, time):
    return np.append(state[time:], state[:time])

def unview(state, time):
    return np.append(state[-time:], state[:-time])

class PantheonLine(MultiAgentEnv):
    def __init__(self):
        super().__init__(ego_ind=0, n_players=2)

        # self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER, NUM_SPACES + 2 * BUFFER, TIME])
        self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.state_ind = -1

        # self.n_reset()

    def update_states(self):
        self.ego_state[self.current_time] = self.state[0] + BUFFER
        self.alt_state[self.current_time] = self.state[1] + BUFFER

    def get_mask(self):
        return np.array([1] * len(VALID_MOVES), dtype=bool)

    def get_full_obs(self):
        ego = view(self.ego_state, self.current_time)
        alt = view(self.alt_state, self.current_time)
        my_obs = np.append(ego, np.append(alt, self.current_time))
        ot_obs = np.append(alt, np.append(ego, self.current_time))
        return (my_obs, my_obs, self.get_mask()), (ot_obs, ot_obs, self.get_mask())

    def n_step(self, actions):
        ego_action = VALID_MOVES[actions[0][0]]
        alt_action = VALID_MOVES[actions[1][0]]

        self.state += np.array([ego_action, alt_action])
        self.current_time -= 1
        self.update_states()

        done = (self.current_time == 0)
        reward = 1.0 if self.state[0] == self.state[1] else -abs(self.state[0] - self.state[1]) * SCALE
        # reward = -abs(self.state[0] - self.state[1])
        for i in range(2):
            if self.state[i] < 0 or self.state[i] >= NUM_SPACES:
                # self.state[i] = 0
                done = True
                reward = -NUM_SPACES * (self.current_time + 1) * SCALE
        return (0, 1), self.get_full_obs(), (reward, reward), done, {}

    def n_reset(self):
        # print("reset")
        self.state_ind = (self.state_ind + 1) % MAX_STATES
        self.state = generate_state(self.state_ind) #np.random.randint(0, NUM_SPACES, 2)
        self.ego_state = np.zeros(TIME)
        self.alt_state = np.zeros(TIME)
        self.current_time = TIME - 1
        self.update_states()
        return (0, 1), self.get_full_obs()


STATIC_ENV = PantheonLine()
    
def validate_step(states, actions, dones, nextstates, verbose=True):
    STATIC_ENV.n_reset()
    
    numenvs = dones.size(0)

    states = states.cpu().numpy()
    actions = actions.cpu().numpy()
    dones = dones.cpu().numpy()
    nextstates = nextstates.cpu().numpy()

    retval = True
    
    for i in range(numenvs):
        STATIC_ENV.state[0] = states[0][i][0] - BUFFER
        STATIC_ENV.state[1] = states[1][i][0] - BUFFER
        STATIC_ENV.current_time = states[0][i][-1]
        STATIC_ENV.ego_state = unview(states[0][i][:TIME], STATIC_ENV.current_time)
        STATIC_ENV.alt_state = unview(states[1][i][:TIME], STATIC_ENV.current_time)
        _, truenext, _, truedone, _ = STATIC_ENV.n_step(actions[:,i])
        truenext = np.array([truenext[0][0], truenext[1][0]])
        # if truedone:
        #     print("FINISHED EPISODE")
        
        if truedone != dones[i]:
            if verbose:
                print("start state:", states[:, i], i)
                print("action:", actions[:, i])
                print("madrona transition:", nextstates[:, i])
                print("numpy transition:", truenext)
                print(f"DONES mismatch: numpy={truedone}, madrona={dones[i] == 1}")
            retval = False
            # return False
            # pass
        if dones[i]:
            # print("MADRONA DONE", nextstates[i])
            continue

        if not np.all(np.abs(truenext - nextstates[:,i]) == 0):
            if verbose:
                print("start state:", states[:, i], i)
                print("action:", actions[:, i])
                print("madrona transition:", nextstates[:, i])
                print("numpy transition:", truenext)
                print("TRANSITIONS are not equal")
            retval = False
            # return False
            # pass
    # print("All good")
    return retval
