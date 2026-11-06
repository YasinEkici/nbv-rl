"""Strict identity and raw SHA-256 binding; synthetic artifacts only."""
import copy
import json
import numpy as np
import pytest
from nbv import evaluation


@pytest.fixture
def artifact(tmp_path):
    selected = {'sarsa': {'setting': {'alpha': .03, 'lam': .5, 'epsilon': 'constant'}}}
    setting = selected['sarsa']['setting']
    config = dict(method='sarsa', seed=0, phi='linear', features=11, budget=2000000,
                  setting=setting, effective_alpha=.03/11, reward_scale=.01, evaluations=20, tail_fraction=.1)
    row = dict(plan='002', status='canonical', part_split='val', method='sarsa', seed='0',
               run_id='synthetic', env_version='v4', split_version='v1', budget_steps='2000000',
               extra_json=json.dumps(dict(tag='p002', feature_count=11, phi='linear', setting=setting, effective_alpha=.03/11)))
    def write(config_override=None, training_override=None):
        import yaml
        cfg = config | (config_override or {})
        tmp_path.mkdir(exist_ok=True)
        (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        (tmp_path/'provenance.json').write_text(json.dumps(dict(dirty=False, env_version='v4', split_version='v1')))
        training = dict(history=[dict(step=s, weight_norm=0., normalized_cycle_time=1.) for s in range(100000,2000001,100000)],
                        policy_weights=[0.]*11, final_weights=[0.]*11, identity=cfg)
        training.update(training_override or {})
        (tmp_path/'training.json').write_text(json.dumps(training))
        (tmp_path/'episodes.jsonl').write_text('')
        np.savez(tmp_path/'weights.npz', tail=np.zeros(11), final=np.zeros(11))
    write()
    return tmp_path, row, selected, config, write


def rejected(artifact, change):
    from nbv.test_evaluation import checkpoint_complete
    directory, row, selected, config, write = artifact
    write(change)
    assert checkpoint_complete(directory)  # the previous guard accepts the corrupted identity
    from nbv.test_evaluation import checkpoint_matches
    assert not checkpoint_matches(directory, row, selected)


def test_reject_wrong_checkpoint_method(artifact):
    rejected(artifact, {'method':'regq'})


def test_reject_wrong_checkpoint_seed(artifact):
    rejected(artifact, {'seed':9999})


def test_reject_wrong_checkpoint_phi(artifact):
    rejected(artifact, {'phi':'none'})


def test_reject_wrong_checkpoint_setting(artifact):
    rejected(artifact, {'setting':{'alpha':.1,'lam':.9,'epsilon':'linear'}})


def test_reject_wrong_checkpoint_feature_identity(artifact):
    from nbv.test_evaluation import checkpoint_complete
    directory, row, selected, config, write=artifact
    # The row claims the wrong feature set while the finite checkpoint is still 11-dimensional.
    extra=json.loads(row['extra_json']); extra['feature_count']=16
    row['extra_json']=json.dumps(extra)
    assert checkpoint_complete(directory)
    from nbv.test_evaluation import checkpoint_matches
    assert not checkpoint_matches(directory,row,selected)


def test_reject_training_identity_disagreement(artifact):
    from nbv.test_evaluation import checkpoint_matches
    directory,row,selected,config,write=artifact
    write(training_override={'identity':config|{'seed':8}})
    assert not checkpoint_matches(directory,row,selected)


def test_accept_matching_checkpoint_identity(artifact):
    from nbv.test_evaluation import checkpoint_matches
    directory,row,selected,_,_=artifact
    assert checkpoint_matches(directory,row,selected)


@pytest.mark.parametrize('name', ['weights.npz','training.json','config.yaml','episodes.jsonl'])
def test_raw_manifest_rejects_changed_artifact(artifact,name):
    from nbv.test_evaluation import artifact_hashes, verify_artifact_hashes
    directory,*_=artifact
    hashes=artifact_hashes(directory)
    assert all(len(value)==64 for value in hashes.values())
    verify_artifact_hashes(directory,hashes)
    with (directory/name).open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError,match='SHA-256'):
        verify_artifact_hashes(directory,hashes)


def test_reject_changed_validation_trajectory():
    from nbv.test_evaluation import assert_val_reproduction
    e=dict(variant='1_v0',sequence=[0,1],cycle_time=12.,success=True)
    assert_val_reproduction([e],[e])
    for key,value in [('sequence',[1,0]),('cycle_time',13.),('success',False)]:
        with pytest.raises(ValueError,match='trajectory'):
            assert_val_reproduction([e],[e|{key:value}])
