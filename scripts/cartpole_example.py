from envs.cartpole_env import CartpoleMadronaTorch, validate_step
import torch
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, enable assertions to validate correctness")
parser.add_argument("--asserts", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, enable assertions to validate correctness")
args = parser.parse_args()

env = CartpoleMadronaTorch(args.num_envs, 0)
old_state = env.reset()
actions = env.static_actions[:, 0, 0]
num_errors = 0
for iter in range(args.num_steps):
    action = torch.randint_like(actions, high=2)

    next_state, reward, next_done, _ = env.step(action)

    if not validate_step(old_state, action, next_done, next_state, verbose=args.verbose):
        # print(old_state, next_state, next_done)
        num_errors += 1
        assert(not args.asserts)

    old_state = next_state
    # print(iter)
    # time.sleep(1)
    # print(reward)
print("Error rate:", num_errors/args.num_steps)
