import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import pairwise_distances
import seaborn as sns
import umap
from collections import Counter
import argparse

# from penetration_test.plot_results import source_folder

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
# source_folder = "archived_results"
source_folder = "results"

def collect_all_test_path_strings(env):
    folder = f"{source_folder}/testing/tiny-{env}"
    all_strings = []
    for method in methods:
        for i in range(1, 4):
            json_file = os.path.join(folder, f"{method}_testing_{i}.json")
            if not os.path.exists(json_file):
                continue
            with open(json_file, "r") as f:
                data = json.load(f)
                for entry in data:
                    actions = [str(a) for a in entry["actions"]]
                    all_strings.append(" ".join(actions))
    return all_strings

def get_reducer(method):
    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == "umap":
        return umap.UMAP(n_components=2, random_state=42)
    elif method == "isomap":
        return Isomap(n_components=2)
    else:
        return PCA(n_components=2)

def plot_all_methods_per_env(env, env_results, out_dir, dim_method):
    combined_X = np.vstack([env_results[m]["X"] for m in env_results])
    current_idx = 0
    reducer = get_reducer(dim_method)
    reduced_X = reducer.fit_transform(combined_X)

    plt.figure(figsize=(10, 8))
    markers = {"DQN": "o", "Random": "*", "RuleBased": "x", "Constrained_DQN": "D", "Constrained_DQN_LambdaNet": "s", "Constrained_DQN_PureNet": "^"}

    for method in env_results:
        X = env_results[method]["X"]
        n = len(X)
        X_2d = reduced_X[current_idx:current_idx + n]
        current_idx += n
        plt.scatter(X_2d[:, 0], X_2d[:, 1],
                    c=color_dict.get(method, "gray"),
                    marker=markers.get(method, "o"),
                    label=method,
                    alpha=0.6,
                    edgecolor='k',
                    s=40)

    plt.title(f"Average Cluster Visualization ({env})", fontsize=20, fontweight='bold')
    plt.legend(fontsize=18, prop={'weight': 'bold'})
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{dim_method.upper()}_average_{env}_cluster.png"))
    plt.close()

def plot_jaccard_heatmap(method, X, env, out_dir):
    distances = pairwise_distances(X, metric="jaccard")
    similarity = 1 - distances
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap="viridis")
    plt.title(f"{method} Jaccard Similarity Heatmap ({env})", fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{method}_{env}_jaccard_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_method", choices=["pca", "tsne", "umap", "isomap"], default="umap",
                        help="Dimension reduction method")
    args = parser.parse_args()

    for env in envs:
        print(f"\n======== ENV: {env} ========")
        folder_name = f"{source_folder}/testing/tiny-{env}"
        all_test_path_strings = collect_all_test_path_strings(env)
        vectorizer = CountVectorizer(binary=True)
        vectorizer.fit(all_test_path_strings)

        info_dict = {}
        env_result_dict = {}

        for method in methods:
            print("==========", method, "===============")
            test_path_array = []
            test_path_set = set()
            sensitive_list = []
            test_len_list = []
            goals = 0

            for i in range(1, 4):
                json_file = os.path.join(folder_name, f"{method}_testing_{i}.json")
                if not os.path.exists(json_file):
                    continue

                with open(json_file, "r") as f:
                    json_data = json.load(f)

                for entry in json_data:
                    actions = [str(a) for a in entry["actions"]]
                    texts = entry["texts"]
                    path_str = " ".join(actions)

                    if entry["goal"]:
                        goals += 1
                        test_len_list.append(len(actions))
                        test_path_set.add("-".join(actions))
                        sensitive_list.append(actions[-1] + ":" + texts[-1])
                        test_path_array.append(path_str)

            if not test_path_array:
                continue
            X = vectorizer.transform(test_path_array).toarray()
            dbscan = DBSCAN(metric='jaccard', eps=0.1, min_samples=2)
            labels = dbscan.fit_predict(X)
            label_counts = Counter(labels)

            result = {
                "labels": len(labels),
                "label_counts": len(label_counts),
                "goals": goals,
                # "test_num": len(test_path_array),
                # "unique_tests": len(test_path_set),
                # "average_len": np.mean(test_len_list),
                # "hit_rate": len(test_path_set) / step_limits[env] * 100,
                "X": X,
            }
            env_result_dict[method] = result
            info_dict[method] = test_path_set

            print(result)

            # Generate Jaccard heatmap for each method
            plot_jaccard_heatmap(method, X, env, out_dir="cluster_visualizations")

        if env_result_dict:
            plot_all_methods_per_env(env, env_result_dict, out_dir="cluster_visualizations", dim_method=args.dim_method)


