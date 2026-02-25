#python run_random_benchmarks.py --seed 0
#
#python run_dqn_policy.py tiny policies/dqn_tiny.pth -seed 0

#[pca|tsne|umap|isomap]
#python plot_clusters.py --dim_method tsne
#python plot_clusters.py --dim_method pca
#python plot_clusters.py --dim_method umap
python plot_clusters.py --dim_method isomap

python plot_results.py
python testing_analysis.py