"""Delivery reporting refuses stale cross-artifact metadata."""
import copy
import pytest


@pytest.mark.parametrize('damage',['bytes','oracle_counts'])
def test_delivery_refuses_stale_bundle_size_or_different_replay_counts(damage):
    from nbv.visualization.report import verify_delivery
    counts={'A':195,'B':900}
    validation={'complete':True,'counts':counts,'mismatches':[]}
    figures={'replay_validation':counts}
    actual={'total_bytes':100,'files':{'example':100}}
    demo=actual|{'verification':counts}
    verify_delivery(validation,figures,demo,actual)
    damaged=copy.deepcopy(demo)
    if damage=='bytes': damaged['total_bytes']=99
    else: damaged['verification']['B']=899
    with pytest.raises(ValueError,match='delivery'):verify_delivery(validation,figures,damaged,actual)
