"""Public documentation, portable evidence and evaluation-weight integrity."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


def test_public_protocol_preserves_preregistration_without_operational_identifiers():
    import re
    text = Path('docs/PROTOCOL.md').read_text(encoding='utf-8')
    assert not re.search(r'\bD-\d{3}\b|\b(?:plan|slice|exec[- ]plan)\s+\d|\b[0-9a-f]{40}\b|\bENV\s+v\d|\bPROTOCOL\s+v\d', text, re.I)
    for required in ('2026-11-03', '2026-11-06', '2026-11-07', 'H1', 'H-forward', 'H-shaping', '1000', '0.005', '0.02'):
        assert required in text
    assert '## Model limitations (reported with every result)' in text


@pytest.mark.parametrize('name', ['polish_comparison_v1', 'collision_worst_1_v1',
                                   'collision_worst_2_v1', 'collision_worst_3_v1'])
def test_historical_receipts_force_the_manifest_line_endings(name):
    path = f'results/visualization/{name}.json'
    actual = subprocess.check_output(['git', 'check-attr', 'eol', '--', path], text=True)
    assert actual.strip().endswith(': crlf')


def test_manuscript_verifier_rejects_a_changed_result():
    from nbv.manuscript import render, verify
    text, claims = render()
    assert verify(text)['claims'] == len(claims)
    result = next(c for c in claims if c['key'] == 'test.sarsa/linear')
    assert result['text'] in text
    with pytest.raises(ValueError, match='source|match'):
        verify(text.replace(result['text'], '999,999999', 1))


def test_manuscript_reads_no_experiment_directories(monkeypatch):
    from nbv.manuscript import render
    original = Path.open
    def guarded(path, *args, **kwargs):
        if {'outputs', 'cache', 'raw'} & set(path.parts):
            raise AssertionError(f'report accessed experiment material: {path}')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    text, _ = render()
    assert 'Ek: Ön kayıt ve değerlendirme protokolü' in text


def _public_weights(tmp_path):
    from nbv.policy_weights import ARM_NAMES, feature_names
    records = [dict(arm=arm, seed=seed, features=feature_names(), weights=[float(seed)]*11,
                    training_date='2026-11-06')
               for arm in ARM_NAMES.values() for seed in range(10)]
    path = tmp_path/'weights.json'
    path.write_text(json.dumps({'weight_kind':'tail mean', 'policies':records}), encoding='utf-8')
    return path, records


def test_public_weights_load_exact_values_and_reject_wrong_hash(tmp_path):
    from nbv.policy_weights import load_policy
    path, records = _public_weights(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    vector = load_policy(path, records[0]['arm'], 0, expected_sha256=digest)
    assert vector.tolist() == records[0]['weights']
    with pytest.raises(ValueError, match='hash'):
        load_policy(path, records[0]['arm'], 0, expected_sha256='0'*64)


def test_public_weights_refuse_feature_reordering_and_final_weights(tmp_path):
    from nbv.policy_weights import load_policy
    path, records = _public_weights(tmp_path)
    records[0]['features'].reverse()
    path.write_text(json.dumps({'weight_kind':'tail mean', 'policies':records}), encoding='utf-8')
    with pytest.raises(ValueError, match='feature'):
        load_policy(path, records[0]['arm'], 0)
    records[0]['features'].reverse()
    records[0]['final_weights'] = records[0]['weights']
    path.write_text(json.dumps({'weight_kind':'tail mean', 'policies':records}), encoding='utf-8')
    with pytest.raises(ValueError, match='field'):
        load_policy(path, records[0]['arm'], 0)
