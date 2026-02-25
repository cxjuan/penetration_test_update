import json
import os
import numpy as np
from itertools import zip_longest


envs = ["gen", "small", "hard"]
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
step_limits = {"gen": 50000, "small": 100000, "hard": 100000}
source_folder = "results"
# archived_results
def collect_all_test_path_strings(env, method):
    folder = f"{source_folder}/testing/tiny-{env}"
    all_strings = []
    num_runs = 4
    for i in range(1, num_runs):
        json_file = os.path.join(folder, f"{method}_testing_{i}.json")
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
            all_strings.append(data)
    return all_strings

def simple_flexible_average(seqs):
    result = []
    for values_at_t in zip_longest(*seqs, fillvalue=None):
        valid_vals = [v for v in values_at_t if v is not None]
        if valid_vals:
            result.append(np.mean(valid_vals))
    return result

for env in envs:
    print(f"\n======== ENV: {env} ========")
    folder_name = f"{source_folder}/testing/tiny-{env}"
    info_dict = {}
    env_result_dict = {}

    for method in methods:
        print("==========", method, "===============")
        json_data = collect_all_test_path_strings(env, method)


        unique_test_path_list = set()
        sensitive_list = []
        test_len_list = []


        all_test_num = []
        all_unique_tests = []
        all_sensitive_list = []
        all_test_len_list = []
        all_goal_list = []
        all_real_hit_rate = []

        all_path = []
        path_set = set()
        path = []
        goals = []
        for i in range(len(json_data)):
            for entry in json_data[i]:
                actions = [str(a) for a in entry["actions"]]
                texts = entry["texts"]
                path_str = " ".join(actions)

                test_len = len(actions)
                test_path = "-".join(actions)
                path.append(1)

                if entry["goal"] is True:
                    goals.append(1)
                    test_len_list.append(test_len)
                    unique_test_path_list.add(path_str)

                    sensitive_list.append(actions[-1] + ":" + texts[-1])

            all_test_num.append(len(json_data[i]))
            all_path.append(len(path))
            all_goal_list.append(len(goals))
            all_unique_tests.append(len(unique_test_path_list))
            all_test_len_list.append(test_len_list)
            all_real_hit_rate.append(len(unique_test_path_list) / step_limits[env] * 100)
            info_dict[method] = set(unique_test_path_list)


        print("round_test_num:\t", np.mean(all_test_num))
        print("path_num:\t", np.mean(all_path))
        print("goals:\t", np.mean(all_goal_list))
        print("self_unique_paths:\t", np.mean(all_unique_tests))
        print("average testing len:\t", np.mean(all_test_len_list))
        print("real_hit_rates:\t", np.mean(all_real_hit_rate))



    print("=========== joint =============")
    common = list(info_dict["DQN"] & info_dict["Constrained_DQN"] & info_dict["Random"] & info_dict["RuleBased"] & info_dict["Constrained_DQN_LambdaNet"] & info_dict["Constrained_DQN_PureNet"])        # strings found in both
    unique_DQN = list(info_dict["DQN"] - info_dict["Constrained_DQN"] - info_dict["Random"] - info_dict["RuleBased"]- info_dict["Constrained_DQN_LambdaNet"] - info_dict["Constrained_DQN_PureNet"])       # only in a
    unique_Constrained_DQN = list(info_dict["Constrained_DQN"] - info_dict["DQN"] - info_dict["Random"] - info_dict["RuleBased"]- info_dict["Constrained_DQN_LambdaNet"] - info_dict["Constrained_DQN_PureNet"])       # only in b
    unique_Random = list(info_dict["Random"] - info_dict["DQN"] - info_dict["Constrained_DQN"] - info_dict["RuleBased"] - info_dict["Constrained_DQN_LambdaNet"] - info_dict["Constrained_DQN_PureNet"])
    unique_RuleBased = list(info_dict["RuleBased"] - info_dict["Constrained_DQN"] - info_dict["DQN"] - info_dict["Random"] - info_dict["Constrained_DQN_LambdaNet"] - info_dict["Constrained_DQN_PureNet"])
    unique_Constrained_DQN_LambdaNet = list(info_dict["Constrained_DQN_LambdaNet"] - info_dict["DQN"] - info_dict["Random"] - info_dict["RuleBased"] - info_dict["Constrained_DQN"] - info_dict["Constrained_DQN_PureNet"])
    unique_Constrained_DQN_PureNet = list(info_dict["Constrained_DQN_PureNet"] - info_dict["DQN"] - info_dict["Random"] - info_dict["RuleBased"] - info_dict["Constrained_DQN"] - info_dict["Constrained_DQN_LambdaNet"])       # only in b



    print("unique_Random:\t", len(set(unique_Random))/3)
    print("unique_RuleBased:\t", len(set(unique_RuleBased))/3)
    print("unique_DQN:\t", len(set(unique_DQN)) / 3)
    print("unique_Constrained_DQN:\t", len(set(unique_Constrained_DQN))/3)
    print("unique_Constrained_DQN_LambdaNet:\t", len(set(unique_Constrained_DQN_LambdaNet)) / 3)
    print("unique_Constrained_DQN_PureNet:\t", len(set(unique_Constrained_DQN_PureNet)) / 3)
            