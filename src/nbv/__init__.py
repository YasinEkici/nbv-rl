"""NBV-RL: next-best-view scan planning for a robot-mounted line-laser profilometer."""

import os

# Windows workers otherwise reserve large BLAS buffers per core. Import nbv before NumPy;
# explicit user settings take precedence (D-047).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

__version__ = "0.1.0"
