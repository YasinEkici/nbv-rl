"""Check saved-pose URDF/DH alignment and exact immutable VAL playback/audit correspondence."""
import argparse
import nbv
from nbv.visualization.validation import verify

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',default='results/visualization')
    args=parser.parse_args();result=verify(args.output_dir)
    print(f"URDF enabled: {result['urdf_enabled']}; exact recorded VAL cases: {len(result['cases'])}")
