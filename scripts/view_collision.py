"""Display the three deepest D-071 recorded collision paths locally."""
import argparse
import time
import nbv
from nbv.visualization.audit_scene import create_audit_server

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8081)
    args=parser.parse_args();server=create_audit_server(args.port)
    try:
        while True:time.sleep(.25)
    except KeyboardInterrupt:pass
    finally:server.stop()
