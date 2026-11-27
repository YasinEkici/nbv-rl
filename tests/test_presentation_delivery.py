"""Tracked-only link checks and committed presentation evidence gates."""
import json
import hashlib
from pathlib import Path
import subprocess


def test_links_resolve_against_tracked_files_not_local_extras():
    from nbv.visualization.links import broken_links
    texts={'README.md':'[ok](docs/a%20b.md) [bad](local.md) [ext](https://example.com)\n```\n[x](example.md)\n```',
           'docs/a b.md':'[up](../README.md#intro)'}
    assert broken_links(texts,{'README.md','docs/a b.md'})==[('README.md','local.md')]
    assert broken_links(texts,{'README.md','docs/a b.md'},[{'file':'README.md','link':'local.md','reason':'historical'}])==[]


def test_all_tracked_markdown_links_and_narrow_historical_exceptions():
    from nbv.visualization.links import broken_links
    paths=set(subprocess.check_output(['git','ls-files'],text=True,encoding='utf-8').splitlines())
    # Historical result records retain their original links and bytes. The public
    # documentation replaces retired operational documents rather than editing evidence.
    texts={p:Path(p).read_text(encoding='utf-8') for p in paths if p.lower().endswith('.md') and not p.startswith('results/')}
    exceptions=json.loads(Path('results/provenance/link_exceptions_v1.json').read_text(encoding='utf-8'))['exceptions']
    active=[e for e in exceptions if e['file'] in texts]
    assert not broken_links(texts,paths,active)
    actual=set(broken_links(texts,paths))
    assert {(e['file'],e['link']) for e in active}==actual,'Stale or overly broad active link exception'


def test_committed_figure_payloads_styles_text_and_manifests():
    from nbv.visualization.design_export import inspect_committed_exports
    result=inspect_committed_exports()
    assert result['reports']==11 and result['slides']==12
    assert result['paired_difference']==-.019894820622543344
    assert result['missing_glyphs']==[]


def test_new_hashed_artifacts_disable_git_line_ending_conversion():
    paths=[]
    for folder in ('results/figures','results/figures/slides'):
        manifest=json.loads(Path(folder,'manifest_v2.json').read_text(encoding='utf-8'))
        paths.extend(str(Path(folder,p)).replace('\\','/') for p in manifest['files'])
    for mode in ('live','recorded','comparison'):paths.append(f'results/visualization/viewer_{mode}_theme_v2.json')
    values=subprocess.check_output(['git','check-attr','text','--',*paths],text=True,encoding='utf-8').splitlines()
    assert all(line.endswith(': unset') for line in values),'Hashed artifacts must preserve serialized bytes on checkout'


def test_new_figure_manifest_hashes_match_git_index_bytes():
    for folder in ('results/figures','results/figures/slides'):
        manifest=json.loads(Path(folder,'manifest_v2.json').read_text(encoding='utf-8'))
        for name,expected in manifest['files'].items():
            path=folder+'/'+name
            data=subprocess.check_output(['git','show',':'+path])
            assert hashlib.sha256(data).hexdigest()==expected,path
