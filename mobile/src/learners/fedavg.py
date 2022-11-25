from src import protocol
from src.conf import WAIT_TIMEOUT, WAIT_INTERVAL
from src.ml import train_for_x_batches
from src.utils import wait_until

name = "Federated averaging (FedAvg)"
# Server is peer.neighbors[-1]
NB_BATCHES = 1


def train_init(peer, args):
    print(f"Done with init")
    peer.params.exchanges = 0
    return


def train_step(peer, t, args):
    T = t if isinstance(t, range) else [t]
    for t in T:
        if t % 10 == 0:
            peer.log('success', f"Round {t}/{len(T)} :: Local train for 1 epoch / {NB_BATCHES} batch(es)...",
                     remote=False)

        if t > 0:
            wait_until(server_received, WAIT_TIMEOUT * 100, WAIT_INTERVAL * 10, peer, t)
            w_server = peer.V[t - 1][0][1]  # [round][n-message(0 in FL)][(id, W)]
            peer.set_model_params(w_server)
        # Worker
        # train_for_x_epochs(peer, epochs=1)
        train_for_x_batches(peer, batches=NB_BATCHES, evaluate=False)
        msg = protocol.train_step(t, peer.get_model_params())  # not grads
        server = peer.neighbors[-1]
        peer.send(server, msg)

    return


def update_model(peer, w, evaluate=False):
    peer.set_model_params(w)
    if evaluate:
        t_eval = peer.evaluate()
        peer.params.logs.append(t_eval)


def train_stop(peer, args):
    peer.stop()


# ---------- Helper functions -------------------------------------------------

def enough_received(peer, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def server_received(peer, t):
    if t - 1 in peer.V and len(peer.V[t - 1]) == 1:
        return True
    return False
