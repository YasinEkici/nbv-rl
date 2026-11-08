"""The live viewer is opt-in and never a base installation requirement."""
import tomllib
from pathlib import Path


def test_viser_is_only_in_optional_demo_extra():
    config=tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']
    assert not any(d.split('=')[0].startswith('viser') for d in config['dependencies'])
    assert any(d.startswith('viser') for d in config.get('optional-dependencies',{}).get('demo',[]))
