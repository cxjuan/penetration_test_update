import matplotlib.pyplot as plt
import json
import numpy as np
from scipy import stats
import itertools

def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data  # not enough data to smooth
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

envs = ["small"]

methods = ["1e-4", "1e-3", "1e-2"]
color_dict = {"1e-4": "orange", "1e-3": "red", "1e-2": "skyblue"}
step_limits = {
    "gen": 50000,
    "small": 500000
}
legend_properties = {'weight': 'bold'}

for env in envs:
    folder_name = "hyperparameters/tiny-" + env
    return_dict = dict()
    cost_dict = dict()
    step_dict = dict()
    for method in methods:
        json_file = folder_name + "/" + method + ".json"
        with open(json_file, "r") as f:
            json_data = json.load(f)

        # Extract values
        steps = [entry[0] for entry in json_data]
        returns = list(itertools.accumulate([max(0, entry[1]) for entry in json_data]))
        returns = np.array(returns)


        costs = np.array([entry[2] for entry in json_data])
        

        goals = np.array([entry[4] for entry in json_data])
        lambdas = np.array([entry[-1] for entry in json_data])

        #### analysis ####
        print("====================================")
        print(env, method, "goals:", sum(goals), sum(goals) / (step_limits[env] / 1000))
        print(env, method, "success rates:", np.sum(returns > 0) / len(returns))
        print(env, method, "lambdas", stats.pearsonr(costs, lambdas))

        steps = moving_average(steps)
        returns = moving_average(returns)
        costs = moving_average(costs)

        indices = steps < step_limits[env]
        steps = steps[indices]
        returns = returns[indices]
        costs = costs[indices]

        steps[0] = 0
        returns[0] = 0
        costs[0] = 0

        print(env, method, "costs:", np.mean(costs[-10:]))
        print(env, method, "returns:", np.mean(returns[-10:]))

        return_dict[method] = returns
        cost_dict[method] = costs
        step_dict[method] = steps

    # Plot episodic return
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(step_dict[method], return_dict[method], label=method, color=color_dict[method])
    plt.xlabel('Training Step', fontsize=20, fontweight='bold')
    plt.ylabel('Episodic Returns', fontsize=20, fontweight='bold')
    plt.title('Accumulative Episodic Returns', fontsize=24, fontweight='bold')
    plt.grid(True)
    plt.legend(prop=legend_properties, fontsize = 22)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig("hyperparameters/" + env + "_returns_hyper.png")

    # Plot cost
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(step_dict[method], cost_dict[method], label=method, color=color_dict[method])
    plt.xlabel('Training Step', fontsize=20, fontweight='bold')
    plt.ylabel('Average Episodic Costs', fontsize=20, fontweight='bold')
    plt.title('Average Episodic Costs', fontsize=24, fontweight='bold')
    plt.grid(True)
    plt.legend(prop=legend_properties, fontsize = 22)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig("hyperparameters/" + env + "_costs_hyher.png")
