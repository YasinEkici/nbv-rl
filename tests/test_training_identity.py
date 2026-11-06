"""Future artifacts persist identity without modifying legacy output folders."""
def test_future_training_payload_contains_frozen_identity():
    from nbv.experiment import training_payload
    fields=('history','policy_weights','final_weights','episode_seeds','secondary','tail_count')
    result={key:[] for key in fields}
    identity=dict(method='sarsa',seed=0,phi='linear',features=11,budget=2000000,
                  setting={'alpha':.03,'lam':.5,'epsilon':'constant'},effective_alpha=.03/11)
    payload=training_payload(result,identity)
    assert payload['identity']==identity
    assert all(payload[k]==result[k] for k in fields)
