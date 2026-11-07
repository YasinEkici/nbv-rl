"""Synthetic preparation reuse only; never run preparation on real test parts."""
import json
from pathlib import Path
import pytest


def test_point_proof_reuses_existing_directory(tmp_path,monkeypatch):
    from nbv import test_inputs as m
    monkeypatch.setattr(m.parts,'load_yaml',lambda p:{'parts':[{'file_id':1,'split':'train'}]})
    monkeypatch.setattr(m,'_nominal_points',lambda *a:None)
    monkeypatch.setattr(m,'array_content_hash',lambda p:'synthetic-content-sha')
    import shutil
    monkeypatch.setattr(shutil,'copyfile',lambda *a:None)
    destination=tmp_path/'proof'
    first=m.prove_point_determinism(destination)
    second=m.prove_point_determinism(destination)
    assert first['original_content_sha256']==second['regenerated_content_sha256']


@pytest.mark.parametrize('tamper',[False,True])
def test_preparation_second_call_checks_committed_hashes(tmp_path,monkeypatch,tamper):
    from nbv import test_inputs as m
    cached=tmp_path/'nominal.npz'; cached.write_bytes(b'original synthetic bytes')
    inventory={'inputs':[dict(path=str(cached),role='nominal_points',exists=True,size_bytes=cached.stat().st_size)],'missing':[]}
    monkeypatch.setattr(m,'dry_run',lambda:inventory)
    monkeypatch.setattr(m,'prove_point_determinism',lambda destination:{'exact_content_match':True})
    manifest=tmp_path/'manifest.json'
    output=tmp_path/'proof'
    first=m.prepare_missing(point_output=output,manifest_path=manifest)
    manifest.write_text(json.dumps(m.make_input_manifest(inventory)))
    if tamper:
        cached.write_bytes(b'changed')
        with pytest.raises(ValueError,match='SHA-256'):
            m.prepare_missing(point_output=output,manifest_path=manifest)
    else:
        second=m.prepare_missing(point_output=output,manifest_path=manifest)
        assert first['generated']==second['generated']==[]
        assert second['already_ready'] is True
