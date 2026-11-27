"""Confirmed source licenses override display metadata without editing evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('file_id', [271875, 37275, 591247])
def test_published_models_have_owner_confirmed_sharealike_license(file_id):
    entries = json.loads(Path('provenance/part_sources.json').read_text(encoding='utf-8'))
    entry = next(row for row in entries if row['file_id'] == file_id)
    confirmation = entry['source_confirmation']
    assert entry['status'] == 403  # Preserve the unsuccessful automated check.
    assert confirmation['license_id'] == 'CC-BY-SA-4.0'
    assert confirmation['license_url'] == 'https://creativecommons.org/licenses/by-sa/4.0/'
    assert confirmation['source_url'] == entry['url']
    assert confirmation['confirmation_method'] == 'repository owner inspected the model page license link'
    assert confirmation['confirmed_on'] == '2026-11-27'
    from nbv.manuscript import render
    text, _ = render()
    import yaml
    part = next(row for row in yaml.safe_load(Path('configs/split_v1.yaml').read_text(encoding='utf-8'))['parts']
                if row['file_id'] == file_id)
    table_row = next(line for line in text.splitlines() if line.startswith('| ' + part['name'] + ' |'))
    assert '| CC BY-SA 4.0 |' in table_row
    assert confirmation['license_url'] in table_row


def test_license_notice_scopes_derivatives_and_discloses_historical_difference():
    text = Path('docs/DATA_LICENSES.md').read_text(encoding='utf-8')
    assert 'CC BY-SA 4.0' in text and 'MIT' in text
    assert 'PHIL' in text and 'CC BY' in text
    assert 'depo sahibi' in text
    for file_id in (271875, 37275, 591247):
        assert f'../results/demo/part_{file_id}_v0.json' in text


def test_viewer_credits_use_confirmed_license_without_mutating_bundle():
    from nbv.visualization.presentation import attribution_text, brief_credit
    path = Path('results/demo/part_271875_v0.json')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = json.loads(path.read_text(encoding='utf-8'))['attribution']
    original = copy.deepcopy(recorded)
    assert recorded['license_version'] is None and not recorded['share_alike']
    for render in (attribution_text, brief_credit):
        text = render(recorded)
        assert 'CC BY-SA 4.0' in text
        assert 'https://creativecommons.org/licenses/by-sa/4.0/' in text
        assert 'sürüm kayıtlı değil' not in text
    assert recorded == original
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_unconfirmed_models_do_not_gain_an_invented_license_version():
    from nbv.data_licenses import display_attribution
    recorded = dict(file_id=46262, source_url='https://www.thingiverse.com/thing:14130',
                    license='Creative Commons - Attribution - Share Alike',
                    license_version=None)
    assert display_attribution(recorded) == recorded
