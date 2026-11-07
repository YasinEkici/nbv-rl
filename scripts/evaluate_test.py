"""Prepared only: do not run until the user separately authorizes opening the test split."""
import nbv
import argparse
from nbv.test_evaluation import open_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i-understand-this-opens-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='stat-only inventory; no test cache content reads')
    parser.add_argument('--resume-after-incident', type=int, choices=[1])
    args = parser.parse_args()
    if args.dry_run:
        from nbv.test_inputs import dry_run, print_inventory
        print_inventory(dry_run())
    elif args.resume_after_incident is not None:
        from nbv.incident_resume import resume_after_incident
        resume_after_incident(incident=args.resume_after_incident)
    else:
        open_test(understand=args.i_understand_this_opens_test)
