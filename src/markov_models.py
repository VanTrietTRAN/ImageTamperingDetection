"""Classifiers on Markov-DCT features: the group's CNN (PyTorch port of the Keras model) and CPU baselines."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn


class MarkovCNN(nn.Module):
    """Port of the notebook model (146,913 parameters).

    layout="keras" reproduces Keras Reshape((9, 9, 24)) on the flat 1944-d vector, which interleaves
    values of different TPMs inside each 3x3 window. layout="aligned" instead maps each of the
    24 TPMs to its own 9x9 channel, so convolutions see real transition-matrix neighbourhoods.
    """

    def __init__(self, T: int = 4, n_maps: int = 24, layout: str = "keras"):
        super().__init__()
        if layout not in ("keras", "aligned"):
            raise ValueError(layout)
        self.dim, self.n_maps, self.layout = 2 * T + 1, n_maps, layout
        self.features = nn.Sequential(
            nn.Conv2d(n_maps, 32, 3, padding="same"), nn.ReLU(), nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 2, padding="same"), nn.ReLU(), nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
        )
        side = self.dim // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * side * side, 128), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(128, 1))

    def forward(self, x):
        b = x.shape[0]
        if self.layout == "keras":
            x = x.reshape(b, self.dim, self.dim, self.n_maps).permute(0, 3, 1, 2)
        else:
            x = x.reshape(b, self.n_maps, self.dim, self.dim)
        return self.head(self.features(x)).squeeze(1)


def train_cnn(Xtr, ytr, seed=42, layout="keras", val_fraction=0.2, max_epochs=200, batch=32, lr=1e-3,
              patience=10, lr_patience=4, threads=None):
    """Same recipe as the notebook: Keras-style validation_split (last 20% of the shuffled training arrays),
    Adam 1e-3, batch 32, EarlyStopping(val_loss, patience 10, restore best), ReduceLROnPlateau(0.5, patience 4)."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = int(len(Xtr) * (1 - val_fraction))
    Xa, ya = torch.from_numpy(Xtr[:split]).float(), torch.from_numpy(ytr[:split]).float()
    Xv, yv = torch.from_numpy(Xtr[split:]).float(), torch.from_numpy(ytr[split:]).float()
    model = MarkovCNN(layout=layout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=lr_patience - 1,
                                                       threshold=1e-4, threshold_mode="abs")
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xa)))
        for i in range(0, len(Xa), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xa[idx]), ya[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv))
        sched.step(vloss)
        history.append(dict(epoch=epoch + 1, val_loss=vloss, lr=opt.param_groups[0]["lr"]))
        if vloss < best:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs=len(history), best_val_loss=best, fit_seconds=time.time() - t0, history=history)


@torch.no_grad()
def cnn_scores(model, X, batch=4096):
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).float())).numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros(0)


def fit_sklearn(kind: str, Xtr, ytr, seed=42):
    """CPU baselines returning a fitted estimator with predict_proba."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    elif kind == "hgb":
        est = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True,
                                             validation_fraction=0.2, random_state=seed)
    else:
        raise ValueError(kind)
    t0 = time.time()
    est.fit(Xtr, ytr)
    est.fit_seconds_ = time.time() - t0
    return est


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) + sum(b.numel() for n, b in model.named_buffers()
                                                            if n.endswith(("running_mean", "running_var")))
