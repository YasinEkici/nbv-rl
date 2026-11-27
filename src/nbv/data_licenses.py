"""Read-only source confirmations, separate from immutable scientific metadata."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source_confirmation(file_id, root=ROOT):
    """Return only an explicitly recorded source-page confirmation."""
    entries = json.loads((Path(root)/'provenance/part_sources.json').read_text(encoding='utf-8'))
    entry = next((row for row in entries if row['file_id'] == file_id), None)
    return entry.get('source_confirmation') if entry else None


def display_attribution(recorded):
    """Overlay confirmed license fields on a copy; never edit source records."""
    result = dict(recorded)
    confirmation = source_confirmation(recorded.get('file_id'))
    if confirmation is None:
        return result
    if confirmation['source_url'] != recorded['source_url']:
        raise ValueError('license confirmation source does not match model attribution')
    result.update(
        license=confirmation['license_label'],
        license_version=confirmation['license_version'],
        official_specific_license_url=confirmation['license_url'],
        share_alike=True,
        version_status_tr='depo sahibinin kaynak sayfa kontrolü: '+confirmation['confirmed_on'],
    )
    return result
