import copy
import os
import time

import tensorflow as tf
from keras import layers as L
from keras.models import Sequential

from src.ml.pytorch.models import *
from .aggregators import average, median, aksel, krum


def initialize_models(model_name, input_shape, nbr_models=1, same=False):
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mae']
    model = Sequential()
    if model_name == 'RNN':
        model.add(L.SimpleRNN(100, activation='relu', input_shape=input_shape))
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
    return model


def model_fit(peer):
    train = peer.dataset.generator.train
    peer.model.fit(train, epochs=4)
    return peer.model.history.history


def model_inference(peer, one_batch=False):
    log('info', f"Model evaluation...")
    test = peer.dataset.generator.test
    logs = peer.model.evaluate(test, verbose=2)
    log('result', f"Node {peer.id} Inference loss: {logs[0]}, RMSE: {logs[1]} | MAE {logs[2]}")


def train_for_x_epoch(peer, batches=1, evaluate=False):
    for i in range(batches):
        # train for x batches randomly chosen when Dataloader is set with shuffle=True
        batch = next(iter(peer.train))
        # execute one training step
        optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr)
        loss = peer.model.train_step(batch, peer.device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get gradients
        # TODO review store gradients in "peer.grads"
        # grads = []
        # for param in peer.model.parameters():
        #     grads.append(param.grad.view(-1))
        # peer.grads = torch.cat(copy.deepcopy(grads))
    if evaluate:
        return peer.model.evaluate(peer.val, peer.device)

    return None


def evaluate_model(model, dataholder, one_batch=False, device="cpu"):
    return model.evaluate(dataholder, one_batch=one_batch, device=device)


def get_params(model, named=False, numpy=False):
    if named:
        return model.get_named_params(numpy=numpy)
    else:
        return model.get_params(numpy=numpy)


def set_params(model, params, named=False, numpy=False):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.set_params(params, numpy=numpy)


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    grads = torch.stack(grads)
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
