"""Persisted evaluation identity must agree with its newly allocated registry run_id."""
def test_evaluation_config_cannot_keep_a_stale_run_id(tmp_path):
    import yaml
    from nbv.frozen_evaluation import save_evaluation
    e=dict(part=1,variant='1_v0',success=True,normalized_cycle_time=1.,cycle_time=10.,passes_to_95=2,
           coverage_at_5=1.,coverage_at_10=1.,auc_coverage_time=.9,total_joint_motion=2.,repair_passes=0,
           nsc_at_5=1.,nsc_at_10=1.,raw_at_5=.8,raw_at_10=.8)
    row=save_evaluation('greedy_cost',0,[e],{'run_id':'stale'}, {'baseline':True},'test',
                        {'commit':'synthetic','dirty':False,'config_files':{}},str(tmp_path/'registry.csv'),
                        str(tmp_path/'outputs'),'synthetic')
    config=yaml.safe_load((tmp_path/'outputs'/row['run_id']/'config.yaml').read_text())
    assert config['run_id']==row['run_id']
