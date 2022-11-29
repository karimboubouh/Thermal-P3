import src.conf as C
from src.ecobee import load_p2p_dataset
from src.edge_device import edge_devices
from src.learners import ctm, fedavg
from src.ml import initialize_models
from src.network import central_graph, network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    """
        - Get the number of rounds/epochs to get same rmse 
        - if acceptable time use the found numbers otherwise reduce it to acceptable conditions
    """
    args = load_conf(use_cpu=False)
    # Configuration ------------------>
    args.mp = 1
    cluster_id = 0
    season = 'summer'
    args.model = "LSTM"
    args.learner = "fedavg"
    args.epochs = 1  # 5
    args.use_batches = False
    args.batch_size = 64
    args.rounds = 20  # set rounds dynamic depending on the rmse of CL
    C.TIME_ABSTRACTION = "1H"
    C.RECORD_PER_HOUR = 1
    resample = False if C.TIME_ABSTRACTION is None else True

    # Details ------------------------>
    fixed_seed(False)
    exp_details(args)
    # Environment setup -------------->

    dataset, input_shape, homes_ids = load_p2p_dataset(args, cluster_id, season, nb_homes=10)  # , nb_homes=10
    models = initialize_models(args.model, input_shape=input_shape, nbr_models=len(dataset), same=True)
    topology = central_graph(models)
    edge = edge_devices(args, count=1)
    graph = network_graph(topology, models, dataset, homes_ids, args, edge=edge)
    # graph.show_neighbors()
    # graph.show_similarity(ids=True)

    # Federated training ------------->
    train_logs = graph.collaborative_training(learner=fedavg, args=args)

    # Plots -------------------------->
    save(f"fl_logs_{args.num_users}_{args.epochs}", train_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(train_logs, metric='rmse', measure="mean-std")
    print("END.")
