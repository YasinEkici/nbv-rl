"""Report reference inclusion must respect D-044 source status."""
import pytest


def test_readme_references_refuse_unverified_citations(tmp_path):
    from nbv.publication import reference_table
    literature=tmp_path/'literature.md'
    literature.write_text('## good — Author, Title\nStatus: verified-source (2026-11-07)\n\n## bad — Placeholder\nStatus: unverified\n',encoding='utf-8')
    assert 'verified-source' in reference_table(['good'],literature)
    with pytest.raises(ValueError,match='unverified'):
        reference_table(['bad'],literature)
