"""Deterministic inference for an existing linear policy; no training or privileged input."""
import numpy as np
from nbv.features import ActionFeatures


class FrozenLinearPolicy:
    def __init__(self,nominal,weights,k_max=24,target=.95):
        self.weights=np.asarray(weights,dtype=np.float64).copy()
        if self.weights.shape!=(11,) or not np.isfinite(self.weights).all():raise ValueError('finite eleven-feature weights required')
        self.features=ActionFeatures(nominal,k_max,reduced=True,target=target)

    def __call__(self,obs,nominal,mask):
        indices=np.flatnonzero(mask)
        return int(indices[np.argmax(self.features(obs,mask)[indices]@self.weights)]) if len(indices) else None
