import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
from itertools import zip_longest
import os
import sys

# archived_results
source_folder = "results"

def compute_avg_cost_success_over_time(all_costs, all_goals):
    avg_success_costs = []
    for t in range(len(all_costs[2])):
        costs_t = [run[t] for run in all_costs if len(run) > t]
        goals_t = [run[t] for run in all_goals if len(run) > t]
        success_costs = [c for c, g in zip(costs_t, goals_t) if g]
        if success_costs:
            avg_success_costs.append(np.mean(success_costs))
        else:
            avg_success_costs.append(np.nan)  # or 0
    return avg_success_costs


def simple_flexible_average(seqs):
    result = []
    for values_at_t in zip_longest(*seqs, fillvalue=None):
        valid_vals = [v for v in values_at_t if v is not None]
        if valid_vals:
            result.append(np.mean(valid_vals))
    return result

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def load_runs(env, folder, method, num_runs=4):
    all_returns = []
    all_costs = []
    all_steps = []
    summaries = []
    all_goals = []
    all_costs_summary = []
    ALL_goal = []
    ALL_costs = []
    round_costs = {}
    round_goals = {}
    
    for i in range(1, num_runs):
        file_path = os.path.join(folder, f"{method}_{i}.json")
        r_goals = []
        r_cost = []
        with open(file_path, "r") as f:
            json_data = json.load(f)

        steps = [entry[0] for entry in json_data]
        returns = list(itertools.accumulate([max(0,entry[1]) for entry in json_data]))
        costs = [entry[2] for entry in json_data]
        goals = [entry[4] for entry in json_data]
        goals_per_episode = [[1 if g else 0] for g in goals]

        r_goals = goals
        r_cost = costs


        all_steps.append(steps)
        all_returns.append(returns)
        all_costs.append(costs)
        all_goals.append(goals)
        ALL_goal.extend(goals)
        ALL_costs.extend(costs)
        round_costs[i] = np.array(ALL_costs)
        round_goals[i] = np.array(ALL_goal)

    step_limit = max(max(steps) for steps in all_steps if steps)
    mean_returns = simple_flexible_average(all_returns)
    mean_costs = simple_flexible_average(all_costs)
    mean_costs = moving_average(mean_costs)
    return_steps = np.linspace(0, step_limit, num=len(mean_returns))
    cost_steps = np.linspace(0, step_limit, num=len(mean_costs))
    avg_cost_success_plt = compute_avg_cost_success_over_time(all_costs, all_goals)

    all_goals = np.array(ALL_goal)
    all_costs_summary = np.array(ALL_costs)
    success_mask = all_goals == True
    fail_mask = all_goals == False

    success_rate = np.mean(success_mask)
    avg_cost_all = np.mean(all_costs_summary)
    avg_cost_success = np.mean(all_costs_summary[success_mask]) if np.any(success_mask) else float('nan')
    avg_cost_fail = np.mean(all_costs_summary[fail_mask]) if np.any(fail_mask) else float('nan')
    std_cost = np.std(all_costs_summary)


    summaries.append({
        "env": env,
        "method": method,
        "success_rate": success_rate,
        "avg_cost_all": avg_cost_all,
        "avg_cost_success": avg_cost_success,
        "avg_cost_fail": avg_cost_fail,
        "std_cost": std_cost,
    })

    return summaries, mean_returns, return_steps, mean_costs, cost_steps, avg_cost_success_plt


envs = [
         "gen",
         "small",
         "hard",
         # "small", "medium"
         ]
standard_env = ["small", "medium"]

methods = [
    "Random", "RuleBased", "DQN",
    "Constrained_DQN",
           "Constrained_DQN_LambdaNet",
           "Constrained_DQN_PureNet",
]
color_dict = {"Random": "green", "RuleBased": "blue", "DQN": "orange", "Constrained_DQN": "red",
              "Constrained_DQN_LambdaNet":"fuchsia",
              "Constrained_DQN_PureNet":"purple",
              }


# for env in envs:
#     folder_name = f"{source_folder}/data/tiny-{env}"

for env in envs:
    folder_name = f"{source_folder}/data/tiny-{env}"

    plt.figure(figsize=(8, 6))

    for method in methods:
        summaries, mean_returns, return_steps, _, _, _ = load_runs(env,folder_name, method)
        print(summaries)
        print('-------------')
        plt.plot(return_steps, mean_returns, label=method, color=color_dict[method])
    plt.xlabel('Training Step', fontsize=20, fontweight='bold')
    plt.ylabel('Episodic Returns', fontsize=20, fontweight='bold')
    plt.title(f'Accumulative Episodic Returns on {env}', fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.legend(fontsize=18, prop={'weight': 'bold'})
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{source_folder}/{env}_returns.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for method in methods:
        _, _, _, mean_costs, cost_steps, _ = load_runs(env,folder_name, method)
        plt.plot(cost_steps, mean_costs, label=method, color=color_dict[method])
    plt.xlabel('Training Step', fontsize=20, fontweight='bold')
    plt.ylabel('Episodic Costs', fontsize=20, fontweight='bold')
    plt.title(f'Average Episodic Costs on {env}', fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.legend(fontsize=18, prop={'weight': 'bold'})
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{source_folder}/{env}_costs.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for method in methods:
        _, _, _, _, cost_steps, avg_cost_success = load_runs(env,folder_name, method)
        episode_indices = np.arange(len(avg_cost_success))
        plt.plot(episode_indices, avg_cost_success, label=method, color=color_dict[method])
    plt.xlabel('Successful Step', fontsize=20, fontweight='bold')
    plt.ylabel('Episodic Costs', fontsize=20, fontweight='bold')
    plt.title(f'Average Successful Episodic Costs on {env}', fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.legend(fontsize=18, prop={'weight': 'bold'})
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{source_folder}/{env}_costs_succeed.png")
    plt.close()
