"""Export the approved Plan 004 VAL viewer bundle; never opens the test holdout."""
import json
import nbv
from nbv.visualization.demo import export

if __name__=='__main__': print(json.dumps(export(),indent=2))
