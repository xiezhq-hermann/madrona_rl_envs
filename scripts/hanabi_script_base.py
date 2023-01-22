from envs.hanabi_env import HanabiMadrona, PantheonHanabi, config_choice

from pantheonrl_extension.vectorenv import SyncVectorEnv

import torch
import time


def get_runtime(use_madrona: bool, use_cpu: bool, num_envs: int, num_steps: int, gpu_id=0, config='full'):
    han_conf = config_choice[config]
    if use_madrona:
        env = HanabiMadrona(num_envs, gpu_id, False, han_conf, use_cpu)
    else:
        env = SyncVectorEnv(
            [lambda: PantheonHanabi(han_conf) for _ in range(num_envs)],
            device=torch.device('cuda', gpu_id)
        )

    start_time = time.time()
    state = env.n_reset()
    actions = torch.zeros((2, num_envs, 1), dtype=int).to(device=env.device)

    for iter in range(num_steps):
        for i in range(2):
            # need to choose valid action (undefined behavior otherwise)
            logits = torch.rand(num_envs, env.action_space.n).to(device=env.device)
            logits[torch.logical_not(state[i].action_mask)] = -float('inf')
            actions[i, :, 0] = torch.max(logits, dim=1).indices

        state, _, _, _ = env.n_step(actions)

    return time.time() - start_time

if __name__ == '__main__':
    print("Time (seconds) for 1000 environments, 50 steps:")
    print("Madrona GPU:", get_runtime(True, False, 1000, 50))
    print("Madrona CPU:", get_runtime(True, True, 1000, 50))
    print("Baseline (single-threaded hanabi_env):", get_runtime(False, False, 1000, 50))
