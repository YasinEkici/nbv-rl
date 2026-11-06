"""A feature diagnostic must not pretend to evaluate the named method's policy."""
import numpy as np


def test_diagnostic_summary_serializes_all_policy_metrics_as_not_applicable():
    from nbv.diagnostics import diagnostic_only_summary
    from nbv.evaluation import registry_row
    summary=diagnostic_only_summary()
    keys=('success_rate','cycle_time_to_95_s','passes_to_95','coverage_at_5','coverage_at_10','auc_coverage_time')
    assert all(np.isnan(summary[key]) for key in keys)
    row=registry_row('diagnostic','sarsa_diagnostic','synthetic','val',1000,summary,
                     {'diagnostic_only':True},'v4','v1','synthetic',budget_steps=5000)
    assert all(row[key]=='' for key in keys)
