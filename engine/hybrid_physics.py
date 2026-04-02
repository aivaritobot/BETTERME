from __future__ import annotations

"""MEJORA GOD: Residual Physics NN (opt-in).

Este módulo se importa condicionalmente desde engine/physics.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # pragma: no cover - opcional en runtime
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class ResidualTrainSample:
    x: list[float]
    y: float


class ResidualPhysicsMLP(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class HybridPhysicsResidual:
    def __init__(self, model_path: str = "models/hybrid_residual.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.enabled = torch is not None and nn is not None
        if self.enabled:
            self.model = ResidualPhysicsMLP(in_dim=6)
            if self.model_path.exists():
                state = torch.load(self.model_path, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()

    def predict_residual_deg(self, features: list[float]) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            out = self.model(x).cpu().numpy().reshape(-1)[0]
        return float(np.clip(out, -25.0, 25.0))

    def train_from_spins(
        self,
        samples: list[ResidualTrainSample],
        epochs: int = 120,
        lr: float = 1e-3,
    ) -> dict:
        if not self.enabled or self.model is None or len(samples) < 12:
            return {"trained": False, "reason": "torch no disponible o pocos datos"}

        x = torch.tensor([s.x for s in samples], dtype=torch.float32)
        y = torch.tensor([[s.y] for s in samples], dtype=torch.float32)

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for _ in range(epochs):
            pred = self.model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        return {"trained": True, "loss": float(loss.item()), "samples": len(samples)}
