"""Measure installed categorical colors and write versioned runtime tokens."""
import nbv
from nbv.visualization.palette import generate
if __name__=='__main__':
    report,theme=generate()
    print('Selected seven:',report['selected_seven'])
    print('Main trio:',report['selected_triple'])
    print('Weakest selected pair:',report['weakest_selected_pair'])
