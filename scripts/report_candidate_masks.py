"""Read-only train/VAL candidate audit; never read the held-out test caches."""
import nbv
from nbv.visualization.candidate_report import audit

if __name__=='__main__':
    r=audit()
    print(r['scope'])
    print({k:v for k,v in r.items() if k.startswith('all_allowed')})
