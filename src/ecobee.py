import copy
import os
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf
from netCDF4 import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import src.conf as C
from src.helpers import Map, timeit
from src.plots import plot_clusters
from src.utils import log



def get_ecobee(force=False, n_clusters=6, get_season=None, get_cluster=None):
    if _process(force):
        # load data
        data = _load_ecobee()
        # get season data
        seasons = {'winter': C.WINTER, 'spring': C.SPRING, 'summer': C.SUMMER, 'autumn': C.AUTUMN}
        season_data = {}
        for name, season in seasons.items():
            season_data[name] = _get_cleaned_season(data, season)
        # cluster homes
        clusters = _cluster_homes(season_data, K=n_clusters)

        # save season data
        dataset = Map()
        log('info', "Getting season data...")
        for n, s in season_data.items():
            dataset[n] = save_season(s, n, clusters)

        if get_season:
            if get_cluster:
                return dataset[get_season][get_cluster]
            else:
                return dataset[get_season]
        return dataset
    else:
        # read and return data for a given cluster and season
        log('info', f"Using already processed Ecobee dataset.")
        if get_season:
            if get_cluster:
                return read_ecobee_cluster(get_cluster, get_season)
            else:
                return read_ecobee_season(get_season)
        return read_processed_dataset()


def read_ecobee_cluster(cluster_id=0, season=None):
    seasons = ['winter', 'spring', 'summer', 'autumn']
    cluster = Map()
    if isinstance(season, str):
        if season.lower() not in seasons:
            log('error', f'Wrong season name: {season}, leave empty for all seasons or pick on from {seasons}')
            exit()
        else:
            seasons = [season]
    elif isinstance(season, list):
        seasons = season

    folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cluster_id}")
    for s in seasons:
        file = os.path.join(folder, f"{s}.csv")
        cluster[s] = pd.read_csv(file, sep=',', index_col='time', parse_dates=True).sort_index()
    return cluster


def read_ecobee_season(season="summer", K=6):
    if isinstance(K, list):
        clusters = K
    else:
        clusters = list(range(K))
    data = []
    for k in clusters:
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{k}")
        file = os.path.join(folder, f"{season}.csv")
        data.append(pd.read_csv(file, sep=',', index_col='time', parse_dates=True).sort_index())

    return pd.concat(data).sort_index()


def read_processed_dataset(K=6):
    seasons = ['winter', 'spring', 'summer', 'autumn']
    dataset = Map(dict.fromkeys(seasons, [None] * K))
    for season in seasons:
        for k in range(K):
            folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{k}")
            file = os.path.join(folder, f"{season}.csv")
            dataset[season][k] = pd.read_csv(file, sep=',', index_col='time', parse_dates=True).sort_index()
    return dataset


def prepare_ecobee(dataset, season="summer", abstraction=True, normalize=True, ts_input=24 * C.RECORD_PER_HOUR,
                   batch_size=1):
    """Expect dataset to represent a season within a cluster"""
    if not isinstance(dataset, pd.DataFrame):
        log('error', f"Provided dataset must be a pandas dataframe")
        exit()
    dataset = _clean_empty_in_temp(dataset)
    if abstraction:
        dataset = dataset.resample(C.TIME_ABSTRACTION).mean()
    X_train, X_test, y_train, y_test = _train_test_split(dataset, season, normalize)
    traing, testg = _timeseries_generator(X_train, X_test, y_train, y_test, length=ts_input, batch_size=batch_size)
    data = Map()
    data['X_train'] = X_train
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['generator'] = Map({'train': traing, 'test': testg})
    return data


# ------------------------- Local functions -----------------------------------

def _process(force=False):
    if force:
        return True
    elif os.path.isdir(C.DATA_CLUSTERS_DIR) or os.path.isdir(C.DATA_SIMILARITY_DIR):
        return False
    return True


@timeit
def _load_ecobee():
    log('info', "Loading Ecobee dataset ...")
    data = {}
    for filename in os.listdir(C.DATA_DIR):
        if filename.endswith('.nc'):
            f = Dataset(os.path.join(C.DATA_DIR, filename), "r", format="NETCDF4")
            key = f.input_files.partition("2017")[2][:2]
            data[key] = {
                'id': f.variables['id'][:],
                'time': f.variables['time'][:],
                'state': f.variables['State'][:],
                'in_temp': f.variables['Indoor_AverageTemperature'][:],
                'in_cool': f.variables['Indoor_CoolSetpoint'][:],
                'in_heat': f.variables['Indoor_HeatSetpoint'][:],
                'in_hum': f.variables['Indoor_Humidity'][:],
                'out_temp': f.variables['Outdoor_Temperature'][:],
                'out_hum': f.variables['Outdoor_Humidity'][:],
                # 'mode': f.variables['HVAC_Mode'][:], # only a masked value: -9999
            }
            f.close()
    log('success', f"Data loaded successfully!")
    return data


@timeit
def _get_cleaned_season(dataset, season, cross_seasons=True):
    log('info', f"Generating season dataset for season {season}")
    data = copy.deepcopy(dataset)
    assert len(season) == 3
    # clean data
    # find users not present in the whole season
    if cross_seasons:
        months = C.WINTER + C.SPRING + C.SUMMER + C.AUTUMN
    else:
        months = season
    ll = list(range(len(months)))
    comb = list(combinations(ll, 2)) + list(combinations(np.flip(ll), 2))
    unique = []
    for mi, mj in comb:
        id_diff = np.setdiff1d(data[months[mi]]['id'], data[months[mj]]['id'])
        unique = np.concatenate((unique, id_diff), axis=0)
    unique = list(set(unique.flatten()))
    print(f"Found {len(unique)} users not sharing data across all {'year' if cross_seasons else 'season'}!")
    # remove users not present in the whole season from dataset
    for m in season:
        # Clean states
        for i in range(data[m]['state'].shape[0]):
            state = next(s for s in data[m]['state'][i] if len(s) > 1)
            data[m]['state'][i][data[m]['state'][i] == ''] = state
        # get unwanted indices
        unwanted = []
        for idk, idv in enumerate(data[m]['id']):
            if idv in unique:
                unwanted.append(idk)
        keys = list(data[m].keys())
        keys.remove('time')
        old_shape = new_shape = None
        for key in keys:
            old_shape = data[m][key].shape
            data[m][key] = np.delete(data[m][key], unwanted, axis=0)
            new_shape = data[m][key].shape
        print(f"Month {m}; Removed {len(unwanted)} unwanted homes; dataset went from {old_shape} to {new_shape}")

    # Concatenate all data of the season
    season_data = {}
    for m in season:
        for key in data[m].keys():
            if key in season_data and key != "id":
                season_data[key] = np.hstack([season_data[key], data[m][key]])
            else:
                season_data[key] = data[m][key]

    print(f"Season data shape for ids is: {season_data['in_temp'].shape}")

    return season_data


def _cluster_homes(data, filename="meta_data.csv", K=6, plot=False):
    log('info', f"Clustering homes depending on {C.META_CLUSTER_COLUMNS[1:]}...")
    # load metadata
    meta_file = os.path.join(C.DATA_DIR, filename)
    df = pd.read_csv(meta_file, usecols=C.META_CLUSTER_COLUMNS)
    meta = df.to_numpy()
    # get home ids
    home_ids = set()
    for season in data.values():
        home_ids.update(season['id'])
    _, indices, _ = np.intersect1d(meta[:, 0], list(home_ids), return_indices=True)
    meta_abs = meta[indices][:, 1:]
    meta_ids = meta[indices][:, 0]
    scaler = MinMaxScaler()
    scaled_meta = scaler.fit_transform(meta_abs)
    kmeans = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = kmeans.fit_predict(scaled_meta)
    clusters_ids = {}
    log("success", f"Clustering finished for {len(indices)} homes.")
    for k in range(K):
        clusters_ids[k] = meta_ids[y_km == k]
        log('', f"Cluster {k} has {len(clusters_ids[k])} homes.")
    if plot:
        plot_clusters(scaled_meta, y_km, kmeans, K)

    return clusters_ids


# @timeit
def save_season(season, name, clusters):
    season_clusters = {}
    for k, cids in clusters.items():
        log('', f"Cluster {k} has {len(cids)} homes in {name} dataset")
        c = {}
        _, indices, _ = np.intersect1d(season['id'], cids, return_indices=True)
        for key, value in season.items():
            if key == "time":
                c[key] = value
            else:
                c[key] = value[indices]
            if key == 'id':
                log('', f"ID SHAPE: {c[key].shape} over {value.shape}")
        season_clusters[k] = c
    # save season data in clusters
    df_season = {}
    log('info', f"Saving clustered {name} datasets ...")
    for ck, cv in season_clusters.items():
        cdf = _cluster2df(cv)
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{ck}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        cfile = os.path.join(folder, f"{name}.csv")
        log('', f"Saving {name} dataset of cluster {ck} to path: {cfile}...")
        cdf.to_csv(cfile, sep=',')
        df_season[ck] = cdf
    return df_season


def _cluster2df(cluster, resample=False):
    df = pd.DataFrame()
    df['time'] = np.tile(cluster['time'], len(cluster['id']))
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, unit='s')
    for ckey, cvalue in cluster.items():
        if ckey in C.DF_CLUSTER_COLUMNS:
            df[ckey] = cvalue.ravel()
    if resample:
        df = df.resample(C.TIME_ABSTRACTION).mean()
    df = df.reindex(columns=C.DF_CLUSTER_COLUMNS)

    return df


def _clean_empty_in_temp(data):
    # TODO Replace with mean(prev, next)
    zeros = (data['in_temp'] == 0).sum()
    percent_diff = round(100 - ((data.size - zeros) / data.size) * 100, 2)
    print(f"Cluster dataset has {zeros} out of {data.size} rows with zeros ({percent_diff}%)\nCleaning...")
    data = data[data['in_temp'] != 0]
    return data


def _train_test_split(data, season, normalize=True):
    break_date = {'winter': "2017-02-15", 'spring': "2017-05-15", 'summer': "2017-08-15", 'autumn': "2017-11-15"}
    # history of target is also part of the training set
    X_train = data[:break_date[season]]
    X_test = data[break_date[season]:]
    y_train = X_train[['in_temp']]
    y_test = X_test[['in_temp']]
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = X_train[:, 5].reshape(-1, 1)
        y_test = X_test[:, 5].reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def _timeseries_generator(X_train, X_test, y_train, y_test, length, batch_size=1):
    TG = tf.keras.preprocessing.sequence.TimeseriesGenerator
    train_generator = TG(X_train, y_train, length=length, batch_size=batch_size)
    Xt = np.vstack((X_train[-length:], X_test))
    yt = np.vstack((y_train[-length:], y_test))
    test_generator = TG(Xt, yt, length=length, batch_size=1)

    return train_generator, test_generator


if __name__ == '__main__':
    pass
