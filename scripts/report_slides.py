"""Generate separate 16:9 presentation figures; frozen report figures are not rewritten."""
import nbv
import argparse
from nbv.visualization import prepare_export_workspace

if __name__=='__main__':
    prepare_export_workspace()
    from nbv.visualization.design_figures import generate,SLIDES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',action='append',choices=tuple(SLIDES))
    parser.add_argument('--output-dir',help='Fresh output destination')
    parser.add_argument('--version',default='v2')
    args=parser.parse_args()
    generate(slides=True,figures=args.figure,output_dir=args.output_dir,version=args.version)
