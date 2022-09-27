import copy
import os

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import layers as L
from keras.models import Sequential
from keras.saving.save import load_model
from tqdm import tqdm
from .aggregators import average, median, aksel, krum
from ...helpers import Map
from ...utils import log
from tqdm.keras import TqdmCallback


def initialize_models(model_name, input_shape, cpu=False, nbr_models=1, same=False):
    models = []
    if same:
        # Initialize all models with same weights
        model, custom_metrics = build_model(model_name, input_shape)
        if nbr_models == 1:
            models.append(model)
        else:
            model_file = f"./{model_name}.model"
            model.save(model_file)
            for i in range(nbr_models):
                models.append(load_model(model_file, custom_objects=custom_metrics))
    else:
        # Independent initialization
        for i in range(nbr_models):
            model, _ = build_model(model_name, input_shape, cpu)
            models.append(model)

    return models


def build_model(model_name, input_shape):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
    custom_metrics = {'mpe_metric': mpe_metric, 'me_metric': me_metric}
    metrics = [rmse, mape, 'mae', mpe_metric, me_metric]
    model = Sequential()
    if model_name == 'RNN':
        model.add(L.SimpleRNN(100, activation='relu', input_shape=input_shape))
    elif model_name == 'LSTM':
        model.add(L.LSTM(100, activation='relu', input_shape=input_shape))
        # if cpu:
        #     model.add(L.LSTM(100, activation='relu', input_shape=input_shape))
        # else:
        #     model.add(L.CuDNNLSTM(100, input_shape=input_shape))
    elif model_name == 'DNN':
        model.add(L.Dense(100, activation='relu', input_shape=(input_shape[1],)))
    elif model_name == 'BNN':
        raise NotImplementedError()
    else:
        exit('Error: Unrecognized model')
    # model.add(L.Dropout(0.2))
    model.add(L.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=metrics)
    return model, custom_metrics


def model_fit(peer, tqdm_bar=False):
    train = peer.dataset.generator.train
    val = peer.dataset.generator.test
    if tqdm_bar:
        # , validation_data = val
        peer.model.fit(train, epochs=peer.params.epochs, batch_size=peer.params.batch_size, verbose=0,
                       callbacks=[TqdmCallback(verbose=2)])
    else:
        # , validation_data=val
        peer.model.fit(train, epochs=peer.params.epochs, validation_data=val, batch_size=peer.params.batch_size,
                       verbose=1)
    history = peer.model.history.history
    h = list(history.values())
    log('result',
        f"Node {peer.id} Train MSE: {h[0][-1]:4f}, RMSE: {h[1][-1]:4f} | MAPE {h[2][-1]:4f} | MAE {h[3][-1]:4f}")

    return history


def train_for_x_epoch():
    raise NotImplementedError()


def train_for_x_batches(peer, batches=1, evaluate=False, use_tqdm=True):
    h1 = Map({'loss': [], 'rmse': [], 'mape': [], 'mae': []})
    h2 = None
    T = tqdm(range(batches), position=0) if use_tqdm else range(batches)
    for _ in T:
        train = peer.dataset.generator.train
        batch = np.random.choice(len(train), 1)
        X, y = train[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
        if h[1] > 1:
            log('error', f"{peer} | h={h} | y={y}, batch={batch}")
        h1.loss.append(h[0])
        h1.rmse.append(h[1])
        h1.mape.append(h[2])
        h1.mae.append(h[3])

    if evaluate:
        test = peer.dataset.generator.test
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h2 = peer.model.test_on_batch(X, y, reset_metrics=False, return_dict=True)
        if h2[1] > 1:
            log('error', f"{peer} | h={h} | y={y}, batch={batch}")

    return Map({'train': h1, 'test': h2})


def model_inference(peer, batch_size=16, one_batch=False):
    test = peer.dataset.generator.test
    if one_batch:
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = peer.model.evaluate(test, verbose=0, batch_size=batch_size)
    history = Map({'loss': h[0], 'rmse': h[1], 'mape': h[2], 'mae': h[3]})
    one = "[^]" if one_batch else "[*]"
    log('result', f"Node {peer.id} Inference {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAPE: {h[2]:4f} | MAE {h[3]:4f}")
    return history


def evaluate_model(peer, one_batch=False):
    test = peer.dataset.generator.test
    if one_batch:
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = peer.model.evaluate(test, verbose=0)

    return {'val_loss': h[0], 'val_rmse': h[1], 'val_mape': h[2], 'val_mae': h[3]}


def evaluate_home(home_id, model, generator, batch_size=16, one_batch=False, dtype="Test "):
    if one_batch:
        batch = np.random.choice(len(generator), 1)
        X, y = generator[batch]
        h = model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = model.evaluate(generator, verbose=0, batch_size=batch_size)
    one = "[^]" if one_batch else "[*]"
    history = Map({'loss': h[0], 'rmse': h[1], 'mape': h[2], 'mae': h[3]})
    log('result', f"Home {home_id} || {dtype} {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAPE: {h[2]:4f} | MAE {h[3]:4f}")
    return history


def get_params(model, named=False, numpy=False):
    if named:
        return {layer.name: layer.get_weights() for layer in model.layers}
    else:
        return model.get_weights()


def set_params(model, params, named=False, numpy=None):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.set_weights(params)


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    # grads = torch.stack(grads)
    if peer.params.gar == "average":
        return average(grads)
    elif peer.params.gar == "median":
        return median(grads)
    elif peer.params.gar == "aksel":
        return aksel(grads)
    elif peer.params.gar == "krum":
        return krum(grads)
    else:
        raise NotImplementedError()


def mape_metric(y_true, y_pred):
    """mean_absolute_percentage_error metric"""
    return K.mean(K.abs((y_true - y_pred) / y_true)) * 100


def me_metric(y_true, y_pred):
    """mean_error metric"""
    return K.mean(y_true - y_pred)


def mpe_metric(y_true, y_pred):
    """mean_percentage_error metric"""
    return K.mean((y_true - y_pred) / y_true) * 100


def timeseries_generator(X_train, X_test, Y_train, Y_test, length, batch_size=128):
    TG = tf.keras.preprocessing.sequence.TimeseriesGenerator
    train_generator = TG(X_train, Y_train, length=length, batch_size=batch_size)
    Xt = np.vstack((X_train[-length:], X_test))
    yt = np.vstack((Y_train[-length:], Y_test))
    test_generator = TG(Xt, yt, length=length, batch_size=batch_size)
    return train_generator, test_generator
