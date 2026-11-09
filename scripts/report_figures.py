"""Generate Plan 004 static figures without accessing the closed test holdout."""
import nbv
from nbv.visualization.design_figures import generate

if __name__=='__main__':
    result=generate()
    print(f"Generated {len(result)} versioned PNG/PDF/SVG/JSON report sets.")
