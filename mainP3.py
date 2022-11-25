import src.conf as C
from src.ecobee import load_p2p_dataset
from src.edge_device import edge_devices
from src.learners import sp3
from src.ml import initialize_models
from src.network import full_graph, network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save, load

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    cpu = False
    args = load_conf(use_cpu=cpu)
    # =================================
    args.mp = 1
    args.model = "LSTM"
    args.batch_size = 64
    args.epochs = 2  # 5
    args.rounds = 100
    cluster_id = 0
    season = 'summer'
    C.TIME_ABSTRACTION = "1H"
    C.RECORD_PER_HOUR = 1
    # =================================
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    dataset, input_shape, homes_ids = load_p2p_dataset(args, cluster_id, season, nb_homes=10)  # , nb_homes=10
    # build users models
    models = initialize_models(args.model, input_shape=input_shape, nbr_models=len(dataset), same=True)
    topology = full_graph(models)
    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=1)
    # build the network graph
    graph = network_graph(topology, models, dataset, homes_ids, args, edge=edge)
    graph.show_neighbors()
    # graph.show_similarity(ids=False)

    # Phase I: Local Training
    graph.local_training(one_batch=True, inference=True)

    # Phase II: Collaborative training
    train_logs = graph.collaborative_training(learner=sp3, args=args)
    save(f"p3_log_{args.num_users}_{args.epochs}", train_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(train_logs, metric='rmse', measure="mean-std", save_fig=True)
    print("END.")

"""
    # load plots
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    train_logs = load("p3_log_100_10_880.pkl")
    plot_train_history(train_logs, metric='rmse', measure="mean")
    exit()
"""
