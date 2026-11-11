"""Generate Plan 004 static figures without accessing the closed test holdout."""
import nbv
import argparse
from nbv.visualization import prepare_export_workspace

if __name__=='__main__':
    prepare_export_workspace()
    from nbv.visualization.design_figures import generate,REPORTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',action='append',choices=tuple(REPORTS),help='Repeat to select a cache-free subset or specific figures')
    parser.add_argument('--output-dir',help='Fresh output destination; existing versioned files are refused')
    parser.add_argument('--version',default='v2')
    args=parser.parse_args()
    result=generate(figures=args.figure,output_dir=args.output_dir,version=args.version)
    print(f"Generated {len(result)} versioned PNG/PDF/SVG/JSON report sets.")
