"""Generate Plan 004 static figures without accessing the closed test holdout."""
import nbv
from nbv.visualization.figures import generate

if __name__=='__main__':
    result=generate()
    print(f"Generated {result['figure_count']} PNG/PDF/JSON figure triples.")
