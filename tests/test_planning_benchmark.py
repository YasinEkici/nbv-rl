"""Planning latency measures action calls only on a shared allowed state collection."""
import pytest


def test_latency_rejects_test_before_constructing_environment():
    from nbv.planning_benchmark import require_validation
    manifest={'variants':[{'variant_id':'1_v0','split':'val'},{'variant_id':'2_v0','split':'test'}]}
    assert require_validation(['1_v0'],manifest)==['1_v0']
    with pytest.raises(ValueError,match='VAL only'): require_validation(['2_v0'],manifest)


def test_latency_counts_decisions_and_keeps_actions_out_of_the_environment():
    from nbv.planning_benchmark import measure_calls
    seen=[]
    policy=lambda obs,nom,mask:seen.append(obs) or 0
    result=measure_calls(policy,[(1,None,None),(2,None,None)],repeats=3)
    assert result['decisions']==6 and seen==[1,2,1,2,1,2]
    assert result['planning_time_ms_per_decision']>=0
