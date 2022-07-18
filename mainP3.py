from src.edge_device import edge_devices
from src.learners import sp3
from src.ml import get_dataset
from src.ml import initialize_models
from src.network import random_graph, network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    args.epochs = 2
    args.rounds = 500
    # =================================
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=True)
    # set up the network topology || 10 (sigma=0.4) // 100 (sigma=0.9) // 300 (sigma=0.95)
    topology = random_graph(models, rho=0.3)  # 1, 0.6, 0.3, 0.05, 0.01
    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=-1)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args, edge=edge)
    graph.show_neighbors()
    # graph.show_similarity(ids=False)

    # Phase I: Local Training
    graph.local_training(inference=False)

    # Phase II: Collaborative training
    train_logs = graph.collaborative_training(learner=sp3, args=args)
    save(f"p3_log_{args.num_users}_{args.epochs}", train_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(train_logs, metric='accuracy', measure="mean-std")
    print("END.")
