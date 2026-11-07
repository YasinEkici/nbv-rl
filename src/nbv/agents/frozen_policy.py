"""Deployed linear policy: nominal features and argmax, no shaping or learning."""
import numpy as np
from nbv.features import ActionFeatures


class FrozenLinearPolicy:
    def __init__(self,weights,k_max=24,target=.95):
        self.weights=np.asarray(weights).copy()
        self.k_max,self.target=k_max,target
        self.features={}

    def __call__(self,obs,nominal,mask):
        key=id(nominal)
        if key not in self.features:
            self.features[key]=ActionFeatures(nominal,self.k_max,reduced=True,target=self.target)
        idx=np.flatnonzero(mask)
        return int(idx[np.argmax(self.features[key](obs,mask)[idx]@self.weights)]) if len(idx) else None
