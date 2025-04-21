import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product

from models import GCN_QNetwork
from utils import *
from env import *

NUM_EPISODES = 50

# options: rl_agent, random_agent, greedy_agent, naive_patrol
PURSUER_STRATEGY = "rl_agent"  
# options: a_star, potential_fields
EVADER_STRATEGY = "potential_fields"
# options: train_env, robotarium
SIM_TYPE = "train_env"

evader_speeds = [0.33, 0.5, 0.75, 0.9, 1.0]
episodes      = range(1, NUM_EPISODES+1)
pairs         = list(product(evader_speeds, episodes))

# load policy if needed
if PURSUER_STRATEGY == "rl_agent":
    model_file = './checkpoints/policy_ep250.pt'
    policy_net = GCN_QNetwork(in_features=11, hidden_dim=64)
    state_dict = torch.load(model_file)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

# prepare results directory & file
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f"results_{PURSUER_STRATEGY}.csv")

with open(results_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # header
    writer.writerow([
        "episode", "evader_speed",
        "success", "num_caught", "chase_time", "getaway_time"
    ])

    # run all speed×episode combinations
    for es, ep in pairs:
        env = PursuitEvasionEnv(
            random_start=True,
            evader_strat=EVADER_STRATEGY,
            pursuer_strat=PURSUER_STRATEGY
        )
        obs = env.reset(
            random_start=True,
            evader_strat=EVADER_STRATEGY,
            pursuer_strat=PURSUER_STRATEGY
        )

        # simulate up to some max steps
        max_steps = 1600
        end_time = max_steps
        success = False
        for step in range(max_steps):
            actions = {}
            for p in env.pursuers:
                if not p.is_moving:
                    if PURSUER_STRATEGY == "rl_agent":
                        x, edge_index = obs_to_tensors(obs, p.id)
                        with torch.no_grad():
                            q_vals = policy_net(x, edge_index)
                        actions[p.id] = int(q_vals.argmax().item())

                    elif PURSUER_STRATEGY == "random_agent":
                        opts = len(p.graph.nodes())
                        actions[p.id] = np.random.randint(opts)

                    elif PURSUER_STRATEGY == "greedy_agent":
                        opts = list(p.graph.nodes())
                        eps = 0.35
                        unv = [i for i,n in enumerate(opts) if n not in p.visited_nodes]
                        if not unv or np.random.rand() < eps:
                            actions[p.id] = np.random.randint(len(opts))
                        else:
                            idx = np.random.choice(unv)
                            actions[p.id] = idx

                    elif PURSUER_STRATEGY == "naive_patrol":
                        opts = list(p.graph.nodes())
                        cnt = env.patrol_counter[p.id]
                        env.patrol_counter[p.id] = (cnt + 1) % len(env.patrol_landmarks[p.id])
                        landmark = env.patrol_landmarks[p.id][cnt]
                        loc = env.reverse_landmarks[str(landmark)]
                        actions[p.id] = opts.index(loc)

            obs, transitions, done = env.step(actions, evader_speed=es)

            if done:
                end_time = step
                num_caught = sum(1 for p in env.pursuers if p.found_evader)
                success = (num_caught == len(env.pursuers))
                break

            # optional: env.display()

        # compute chase/getaway times
        chase_time   = end_time if success else -1
        getaway_time = -1 if success else end_time

        # write out this episode’s results
        writer.writerow([ep, es, int(success), num_caught, chase_time, getaway_time])
        csvfile.flush()  # ensure it's written to disk
        print(f"Speed {es:.2f} | Ep {ep}/{NUM_EPISODES} → success={success}, time={end_time}")

print(f"All done! Results saved to {results_file}")
