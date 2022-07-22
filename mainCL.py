from src.ecobee import read_ecobee_cluster, prepare_ecobee
from src.p2p import Graph
from src.utils import load_conf, fixed_seed, exp_details

"""
Clustering finished for 954 homes.
    Cluster 0 has 168 homes.
    Cluster 1 has 166 homes.
    Cluster 2 has 206 homes.
    Cluster 3 has 138 homes.
    Cluster 4 has 194 homes.
    Cluster 5 has 82 homes.
"""
if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # Centralized training
    train_logs = Graph.centralized_training(args, cluster_id=0, season='summer')
    # save(f"train_logs_CL", train_logs)
    # info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of Epochs"}
    # plot_train_history(train_logs, metric='accuracy', measure="mean")
    print("END.")
