"""Multi-seed PyTorch MLP for the walk-forward engine."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from factor_model.models.base import BaseModel

# Modest grid: architecture × dropout × weight decay.
MLP_GRID = {
    "hidden": [(32, 16), (64, 32, 16)],
    "dropout": [0.1, 0.3],
    "weight_decay": [1e-3, 1e-2],
}


class _MLPNet(nn.Module):
    def __init__(self, n_in: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _hidden_key(h: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(h, str):
        # "32-16" form from JSON
        return tuple(int(x) for x in h.split("-"))
    return tuple(int(x) for x in h)


class MLPModel(BaseModel):
    name = "mlp"
    param_grid = MLP_GRID
    needs_validation = True

    def __init__(
        self,
        hidden: Sequence[int] | str = (32, 16),
        dropout: float = 0.1,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        max_epochs: int = 40,
        patience: int = 6,
        batch_size: int = 4096,
        n_seeds: int = 5,
        seed: int = 42,
        **kwargs,
    ) -> None:
        hidden_t = _hidden_key(hidden)
        super().__init__(
            hidden=hidden_t,
            dropout=float(dropout),
            weight_decay=float(weight_decay),
            lr=float(lr),
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            n_seeds=int(n_seeds),
            seed=int(seed),
            **kwargs,
        )
        self._models: list[_MLPNet] = []
        self.seed_preds_std_: float | None = None  # set on predict if multi-seed

    def clone(self, **params):
        merged = {**self.params, **params}
        if "hidden" in merged:
            merged["hidden"] = _hidden_key(merged["hidden"])
        return self.__class__(**merged)

    def _train_one(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        seed: int,
    ) -> _MLPNet:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        net = _MLPNet(X.shape[1], self.params["hidden"], self.params["dropout"]).to(device)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        loss_fn = nn.MSELoss()
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)
        best_state = None
        best_val = float("inf")
        wait = 0
        Xv = yv = None
        if X_val is not None and y_val is not None:
            Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)

        for _ in range(self.params["max_epochs"]):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(net(Xv), yv).item())
                if vloss < best_val - 1e-6:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.params["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MLPModel:
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self._models = []
        base = int(self.params["seed"])
        for i in range(int(self.params["n_seeds"])):
            self._models.append(self._train_one(X, y, X_val, y_val, seed=base + i))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("MLPModel.fit must be called first")
        xt = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for m in self._models:
                m.eval()
                preds.append(m(xt).cpu().numpy())
        stacked = np.stack(preds, axis=0)
        self.seed_preds_std_ = float(stacked.std(axis=0).mean())
        return stacked.mean(axis=0)
