"""Read-only allowed-split D-042 collision measurement; never calls a policy."""
import argparse
import nbv
from nbv.collision_audit import check_waypoints,prepare_records,run_audit,verify_query_equivalence,finalize_audit,report_path,combine_audits

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waypoints-only',action='store_true')
    parser.add_argument('--prepare-records',action='store_true')
    parser.add_argument('--verify-query-equivalence',action='store_true')
    parser.add_argument('--finalize',action='store_true',help='Validate ledger and regenerate summaries only; no geometry evaluation')
    parser.add_argument('--measurement-commit',help='Original core commit for a legacy ledger without embedded code provenance')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--scope',choices=['all','val','train-primary','train-final'],default='all')
    parser.add_argument('--warm-cache',help='Hash-verified private geometry snapshots; unchanged 1-degree sampling')
    parser.add_argument('--combine',action='store_true',help='Combine completed VAL/primary train ledgers only; no geometry work')
    parser.add_argument('--include-final',action='store_true')
    parser.add_argument('--shards-per-part',type=int,default=1)
    parser.add_argument('--reuse-complete-parts',action='store_true')
    args=parser.parse_args()
    if args.waypoints_only:
        r=check_waypoints();print(f"Saved waypoint gate passed: {r['checked']} samples")
    elif args.prepare_records:
        r=prepare_records();print(f"Exact recorded traces prepared: {r['records']} episodes")
    elif args.verify_query_equivalence:
        r=verify_query_equivalence(args.workers);print(f"Query parallelism: {r['poses']} poses bit-identical")
    elif args.finalize:
        r=finalize_audit(report_path(args.scope),measurement_commit=args.measurement_commit,warm_manifest=args.warm_cache);print(f"Audit ledger verified: {len(r['episodes'])} episodes")
    elif args.combine:
        r=combine_audits(include_final=args.include_final);print(f"Combined measured episodes: {len(r['episodes'])}")
    else:
        r=run_audit(workers=args.workers,scope=args.scope,warm_cache=args.warm_cache,
                    shards=args.shards_per_part,reuse_parts=args.reuse_complete_parts);print(f"Collision audit complete: {len(r['episodes'])} episodes")
