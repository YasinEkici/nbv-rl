"""Shared report text from the frozen protocol; no numerical results or environments."""
from pathlib import Path


def collision_limitation():
    return ('D-071 diagnoses sampled train/VAL recorded-path collisions in unchecked transfers and between scan poses; '
            'see the [collision audit](collision_audit_v1.md). Frozen quality results and hypotheses are unchanged.')


def model_limitations():
    protocol=Path('docs/PROTOCOL.md').read_text(encoding='utf-8')
    heading='## Model limitations (reported with every result)'
    body=protocol.split(heading+'\n',1)[1].split('\n## ',1)[0].strip()
    note='\n\n'+collision_limitation() if Path('results/collision_audit_v1.json').exists() else ''
    return '\n\n'+heading+'\n\n'+body+note+'\n'


def update_collision_limitation():
    """Insert only the new nonnumeric limitation; never reopen or rerender a test episode."""
    from nbv.publication import replace_block
    path=Path('results/test_v1.md')
    heading='## Model limitations (reported with every result)'
    text=path.read_text(encoding='utf-8')
    prefix,tail=text.split(heading,1)
    # Existing report has the protocol limitations as its final section.
    note=collision_limitation()
    if note not in tail:path.write_text(prefix+heading+tail.rstrip()+'\n\n'+note+'\n',encoding='utf-8')
    readme=Path('README.md').read_text(encoding='utf-8')
    begin='<!-- BEGIN GENERATED: model_limitations -->';end='<!-- END GENERATED: model_limitations -->'
    block=readme.split(begin,1)[1].split(end,1)[0].strip()
    readme_note=note.replace('(collision_audit_v1.md)','(results/collision_audit_v1.md)')
    if readme_note not in block:replace_block('README.md','model_limitations',block+'\n\n'+readme_note)
