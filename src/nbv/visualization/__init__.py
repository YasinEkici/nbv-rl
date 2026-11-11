"""Recorded-result visualization. The real test holdout is registry-only."""


def prepare_export_workspace():
    """Own temporary/cache directories; call before importing Matplotlib in entry points."""
    import os
    from pathlib import Path
    scratch=Path('scratch').resolve();scratch.mkdir(parents=True,exist_ok=True)
    config=scratch/'matplotlib';config.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(config))
    return scratch
