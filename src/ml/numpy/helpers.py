import copy
import time

import tensorflow as tf
from keras import layers as L
from keras.models import Sequential

from .layers import BasicRNN, Activation, Flatten, Dense
from .Model import Model
from src.utils import log
from .aggregators import average, median, aksel, krum
from .utils import flatten_grads, unflatten_grad


def initialize_models(model_name, input_shape, cpu=True, nbr_models=1, same=False):
    print("initialize_models")
    exit()
    models = []
    if same:
        # Initialize all models with same weights
        model = build_model(model_name, input_shape)
        if nbr_models == 1:
            models.append(model)
        else:
            for i in range(nbr_models):
                models.append(copy.deepcopy(model))
    else:
        # Independent initialization
        for i in range(nbr_models):
            models.append(build_model(model_name, input_shape))

    return models


def build_model(model_name, input_shape):
    metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mae']
    model = Sequential()
    if model_name == 'RNN':
        # model.add(L.SimpleRNN(100, activation='relu', input_shape=input_shape))
        architecture = [
            BasicRNN(name='R1', units=10, return_last_step=False),
            Activation(name='A1', method='relu'),
            BasicRNN(name='R2', units=10, return_last_step=False),
            Activation(name='A2', method='relu'),
            Flatten(name='flatten'),
            Dense(name='fc1', units=10),
            Activation(name='A3', method='softmax'),
        ]
        model = Model(name="RNN", input_dim=input_shape).initial(architecture)
    elif model_name == 'LSTM':
        model.add(L.SimpleRNN(100, activation='relu', input_shape=input_shape))
    elif model_name == 'DNN':
        model.add(L.Dense(100, activation='relu', input_shape=(input_shape[1],)))
    elif model_name == 'BNN':
        raise NotImplementedError()
    else:
        exit('Error: Unrecognized model')
    # model.add(L.Dropout(0.2))
    model.add(L.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=metrics)
    model.summary()
    print(model.summary())
    print("--------------------")
    return model


def model_fit(peer):
    peer.model.train(peer.train.dataset, peer.train.targets)
    peer.model.val(peer.val.dataset, peer.val.targets)
    peer.model.test(peer.test.dataset, peer.test.targets)
    history = peer.model.fit(
        lr=peer.params.lr,
        momentum=peer.params.momentum,
        max_epoch=peer.params.epochs,
        batch_size=peer.params.batch_size,
        evaluation=True,
        logger=log
    )
    return history


def train_for_x_epoch(peer, batches=1, evaluate=False):
    # TODO improve FedAvg for numpy
    if peer.model.has_no_data():
        peer.model.train(peer.train.dataset, peer.train.targets)
        peer.model.val(peer.val.dataset, peer.val.targets)
        peer.model.test(peer.test.dataset, peer.test.targets)
    return peer.model.improve(batches, evaluate)


def evaluate_model(model, dataholder, one_batch=False, device=None):
    loss, acc = model.evaluate(dataholder.dataset, dataholder.targets, one_batch=one_batch)
    return {'val_loss': loss, 'val_acc': acc}


def model_inference(peer, batch_size=16, one_batch=False):
    t = time.time()
    loss, acc = peer.model.evaluate(peer.inference.dataset, peer.inference.targets, one_batch, verbose=0)
    o = "1B" if one_batch else "*B"
    t = time.time() - t
    log('result', f"{peer} [{t:.2f}s] {o} Inference loss: {loss:.4f},  acc: {(acc * 100):.2f}%")


def get_params(model, named=False, numpy=None):
    if named:
        return model.named_parameters()
    else:
        return model.parameters


def set_params(model, params, named=False, numpy=None):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.parameters = params


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    flattened = flatten_grads(grads)
    if peer.params.gar == "average":
        r = average(flattened)
    elif peer.params.gar == "median":
        r = median(flattened)
    elif peer.params.gar == "aksel":
        r = aksel(flattened)
    elif peer.params.gar == "krum":
        r = krum(flattened)
    else:
        raise NotImplementedError()
    return unflatten_grad(r, grads[0])
