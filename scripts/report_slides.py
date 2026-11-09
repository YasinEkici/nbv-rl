"""Generate separate 16:9 presentation figures; frozen report figures are not rewritten."""
import nbv
from nbv.visualization.design_figures import generate

if __name__=='__main__':generate(slides=True)
