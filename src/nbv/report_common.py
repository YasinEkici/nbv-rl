"""Shared report text from the frozen protocol; no numerical results or environments."""
from pathlib import Path


def model_limitations():
    protocol=Path('docs/PROTOCOL.md').read_text(encoding='utf-8')
    heading='## Model limitations (reported with every result)'
    body=protocol.split(heading+'\n',1)[1].split('\n## ',1)[0].strip()
    return '\n\n'+heading+'\n\n'+body+'\n'
