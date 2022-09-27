import src.conf as C
from src import plots
from src.helpers import Map
from src.p2p import Graph
from src.plots import box_plot
from src.utils import load_conf, fixed_seed, exp_details, save, load

if __name__ == '__main__':
    # ----------------------¬
    cpu = False
    cluster_id = 4
    season = 'summer'
    # ----------------------
    args = load_conf(cpu=cpu)
    # ----------------------
    args.model = "RNN"
    args.epochs = 4
    args.batch_size = 128
    C.TIME_ABSTRACTION = "15min"
    C.RECORD_PER_HOUR = 4
    resample = False if C.TIME_ABSTRACTION is None else True
    # ----------------------s
    fixed_seed(True)
    exp_details(args)
    # Centralized training
    train_log, homes_logs, predictions = Graph.centralized_training(args, cluster_id, season, resample, predict=False)

    #

    #

    #

    #

    #

    #

    # save(f"CL_logs_{cpu}_cluster_{cluster_id}_{season}_{args.epochs}", [train_log, homes_logs])
    # train_log, homes_logs = load("CL_logs_False_cluster_4_winter_4_620.pkl")
    # plot predictions
    # info = Map({'xlabel': "Time period", 'ylabel': 'Temperature'})
    # plots.plot_predictions(predictions, info)
    # plot box_plot
    # info = {'xlabel': "Epochs", 'ylabel': 'Validation RMSE Loss'}
    # box_plot(train_log, homes_logs, showfliers=False, title=f"Cluster {cluster_id} || Season {season}")
    print("END.")
#

"""
---> Centralized learning of all clusters
    train_logs = {}
    for c in range(6):
        for s in ['spring', 'summer', 'autumn']:
            log('event', f"Training model of cluster {c} for {s} season...")
            logs = Graph.centralized_training(args, cluster_id=c, season=s)
            train_logs[f"{c}_{s}"] = logs
    save(f"CL_logs_CPU_ALL_{args.epochs}_{TIME_ABSTRACTION}", train_logs)
"""
