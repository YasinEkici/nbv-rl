"""Local VAL-only live inference or SHA-verified recorded-bundle playback."""
import argparse
import time
import nbv
from nbv.visualization.live import create_server,validate_headless

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--recorded',action='store_true',help='Play immutable v0/seed-0 bundle only; no environments/checkpoints needed')
    parser.add_argument('--weights',help='Published tail-weight JSON; live mode still needs original VAL caches and records')
    parser.add_argument('--validate-headless',action='store_true',help='Write the fixed-case VAL reproduction diagnostic, without a server')
    args=parser.parse_args()
    if args.validate_headless:
        result=validate_headless()
        print(f"Headless VAL: {len(result['cases'])} cases exactly matched; registry and bundle unchanged.")
        raise SystemExit(0)
    server=create_server(args.port,recorded=args.recorded,weights_path=args.weights)
    print(f'Open http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        while True:time.sleep(.25)
    except KeyboardInterrupt:pass
    finally:server.stop()
