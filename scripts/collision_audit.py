"""Read-only allowed-split D-042 collision measurement; never calls a policy."""
import argparse
import nbv
from nbv.collision_audit import check_waypoints,prepare_records,run_audit,verify_query_equivalence,finalize_audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waypoints-only',action='store_true')
    parser.add_argument('--prepare-records',action='store_true')
    parser.add_argument('--verify-query-equivalence',action='store_true')
    parser.add_argument('--finalize',action='store_true',help='Validate ledger and regenerate summaries only; no geometry evaluation')
    parser.add_argument('--measurement-commit',help='Original core commit for a legacy ledger without embedded code provenance')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.waypoints_only:
        r=check_waypoints();print(f"Saved waypoint gate passed: {r['checked']} samples")
    elif args.prepare_records:
        r=prepare_records();print(f"Exact recorded traces prepared: {r['records']} episodes")
    elif args.verify_query_equivalence:
        r=verify_query_equivalence(args.workers);print(f"Query parallelism: {r['poses']} poses bit-identical")
    elif args.finalize:
        r=finalize_audit(measurement_commit=args.measurement_commit);print(f"Audit ledger verified: {len(r['episodes'])} episodes")
    else:
        r=run_audit(workers=args.workers);print(f"Collision audit complete: {len(r['episodes'])} episodes")
