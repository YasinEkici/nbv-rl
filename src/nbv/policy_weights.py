"""Read-only, exact numerical serialization for published linear policies."""
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from nbv.features import FEATURE_NAMES, REDUCED_INDICES


ARM_NAMES = {
    ('sarsa', 'linear'): 'SARSA · doğrusal potansiyel',
    ('sarsa', 'none'): 'SARSA · şekillendirme yok',
    ('sarsa', 'convex'): 'SARSA · dışbükey potansiyel',
    ('sarsa', 'value_estimate'): 'SARSA · değer tahmini potansiyeli',
    ('regq', 'linear'): 'RegQ · doğrusal potansiyel',
    ('myopic', 'linear'): 'Miyop · doğrusal potansiyel',
}


def feature_names():
    return [FEATURE_NAMES[int(i)] for i in REDUCED_INDICES]


def load_policy(path, arm, seed, *, expected_sha256=None):
    """Validate the complete public file and return the selected binary64 vector."""
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('published weights hash mismatch')
    document = json.loads(payload)
    if set(document) != {'weight_kind', 'policies'}:
        raise ValueError('unexpected public weight document field')
    expected_fields = {'arm', 'seed', 'features', 'weights', 'training_date'}
    expected_keys = {(name, i) for name in ARM_NAMES.values() for i in range(10)}
    vectors = {}
    for row in document['policies']:
        if set(row) != expected_fields:
            raise ValueError('unexpected public weight row field')
        if row['features'] != feature_names():
            raise ValueError('published feature order mismatch')
        key = (row['arm'], row['seed'])
        if type(row['seed']) is not int or key not in expected_keys or key in vectors:
            raise ValueError('unknown or duplicated arm/seed')
        date.fromisoformat(row['training_date'])
        vector = np.asarray(row['weights'], dtype=np.float64)
        if vector.shape != (len(feature_names()),) or not np.isfinite(vector).all():
            raise ValueError('finite feature-aligned weight vector required')
        vectors[key] = vector
    if vectors.keys() != expected_keys:
        raise ValueError('all six arms and ten seeds are required')
    if (arm, seed) not in vectors:
        raise ValueError('unknown published arm/seed')
    return vectors[(arm, seed)].copy()
