"""Lossless demo encoding, export gates, attribution and whole-bundle integrity."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest


def test_typed_arrays_roundtrip_binary64_and_reject_wrong_length_or_nonfinite():
    from nbv.visualization.demo import encode_array,decode_array
    a=np.array([[0.,-0.,np.nextafter(1.,2.)],[1e-100,np.pi,1e100]],dtype='>f8')
    packed=encode_array(a,'mm')
    assert packed['dtype']=='<f8' and packed['byte_order']=='little' and packed['order']=='C'
    assert decode_array(packed).tobytes()==a.astype('<f8').tobytes()
    damaged=packed|{'shape':[7,3]}
    with pytest.raises(ValueError,match='length'): decode_array(damaged)
    with pytest.raises(ValueError,match='finite'): encode_array(np.array([np.nan]),'rad')


def test_packed_visibility_retains_pass_sample_order_and_bit_count():
    from nbv.visualization.demo import pack_visibility,decode_array
    bits=(np.arange(48*19).reshape(48,19)%7)==0
    packed=pack_visibility(bits)
    assert packed['bit_order']=='big' and packed['logical_shape']==[48,19] and packed['bit_count']==48*19
    assert np.array_equal(np.unpackbits(decode_array(packed),axis=1,count=19,bitorder='big'),bits)


def test_attribution_never_guesses_missing_license_version():
    from nbv.visualization.demo import attribution
    source=dict(file_id=1,name='toy',author='author',thing_url='https://example.org/toy',license='Creative Commons - Attribution - Share Alike',share_alike=True)
    item=attribution(source)
    assert item['version_status']=='version not recorded' and item['license_version'] is None
    assert item['official_specific_license_url'] is None and item['share_alike']
    assert item['changes']['variant_points'] and item['changes']['visibility']
    assert 'deviation' not in ' '.join(item['changes']['nominal_mesh']).lower()


def test_episode_export_blocks_unverified_or_changed_sequence_and_keeps_home_separate():
    from nbv.visualization.demo import episode_document
    from nbv.visualization.replay import replay_episode
    from test_visual_replay import record
    from test_env import make_env
    env=make_env();home=env.q_home.tolist()
    r=replay_episode(env,record(),10.,stage='B')
    r.update(split='val',method='sarsa/linear',seed=0,source_run='synthetic')
    out=episode_document(r,home,'fake-full-oracle-manifest-sha')
    assert len(out['steps'])==len(r['steps'])+1 and out['steps'][-1]['kind']=='home'
    assert out['steps'][-1]['cumulative_time_s']==r['summary']['cycle_time']
    assert out['verification']['final_coverage_status']=='derived from replay; consistent with partial oracle'
    bad=copy.deepcopy(r);bad['verification']['exact']=False
    with pytest.raises(ValueError,match='unverified'):episode_document(bad,home,'fake')
    bad=copy.deepcopy(r);bad['summary']['sequence'][0]=99
    with pytest.raises(ValueError,match='sequence'):episode_document(bad,home,'fake')


def test_schema_requires_typed_finite_arrays_and_version():
    from jsonschema import Draft202012Validator,ValidationError
    from nbv.visualization.demo import schema,encode_array
    doc=schema();Draft202012Validator.check_schema(doc)
    validate=Draft202012Validator(doc['$defs']['array']).validate
    validate(encode_array(np.ones((2,3)),'rad'))
    with pytest.raises(ValidationError):validate(encode_array(np.ones(3),'rad')|{'dtype':'float16'})
    with pytest.raises(ValidationError):Draft202012Validator(doc).validate({'kind':'part','schema_version':99})


def test_bundle_budget_counts_manifest_and_integrity_detects_changed_file(tmp_path):
    from nbv.visualization.demo import write_manifest,check_bundle
    (tmp_path/'common.json').write_text('{"schema_version":1}',encoding='utf-8')
    write_manifest(tmp_path,commit='synthetic')
    info=check_bundle(tmp_path)
    assert info['total_bytes']==sum(p.stat().st_size for p in tmp_path.iterdir())
    with pytest.raises(ValueError,match='12000000'):check_bundle(tmp_path,limit=info['total_bytes']-1)
    (tmp_path/'common.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):check_bundle(tmp_path)


def test_export_rejects_test_geometry_before_opening_any_asset(monkeypatch):
    from nbv.visualization.demo import part_document
    from nbv.visualization import assets
    monkeypatch.setattr(assets,'load_scene',lambda *a,**kw:pytest.fail('geometry was opened'))
    with pytest.raises(ValueError,match='VAL'):part_document(178195,[],{},'fake')
