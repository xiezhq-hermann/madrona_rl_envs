from envs.balance_beam_env import BalanceMadronaTorch, validate_step
import torch
import time

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, enable assertions to validate correctness")
parser.add_argument("--asserts", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, enable assertions to validate correctness")

parser.add_argument("--validation", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, validate correctness")
parser.add_argument("--debug-compile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, validate correctness")
args = parser.parse_args()

env = BalanceMadronaTorch(args.num_envs, 0, args.debug_compile)
old_state = env.reset()
actions = env.static_actions
num_errors = 0
start_time = time.time()
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    action = torch.randint_like(actions, high=4)

    next_state, reward, next_done, _ = env.step(action)

    if args.validation and not validate_step(old_state, action, next_done, next_state, verbose=args.verbose):
        num_errors += 1
        assert(not args.asserts)

    old_state = next_state
    # print(iter)
    # time.sleep(1)
    # print(reward)
print("SPS:", args.num_steps / (time.time() - start_time))
if args.validation:
    print("Error rate:", num_errors/args.num_steps)
