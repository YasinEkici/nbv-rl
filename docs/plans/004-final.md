# Plan 004 final presentation corrections

Status: active. Opened: 2026-11-11.
Branch: fix/004-final, from main (explicit delivery exception).
Related decision: D-072. Presentation-only bug fixes; no new scientific decision.

## Scope and preservation

1. Verify every report/presentation CAD marker from drawn objects, then fix the motion plot and publish its corrected version with exact original data fields.
2. Verify every viewer curve's line style against the committed theme, then apply supported Plotly dash values. Stop if the installed chart library does not support them.
3. Verify export from a temporary tracked-only worktree with no scratch directory, using cache-free figures. The producer creates its temporary directories.

Observe red before implementation and pass the full pytest suite before every commit. Source-code and generated-figure commits are separate so the figure records its actual committed renderer. Retain all historical figures/manifests under their original names; publish the one affected figure as v3. Registry, scientific settings/results, demo package, ENV v4 and collision calculations stay unchanged. Test figures read registry metrics only; the holdout remains closed. Never publish remote changes.

## Progress

- Correction 1 red: all 23 report/presentation render cases were drawn; 22 passed and the motion plot failed because its CAD marker was filled. Geometry-only panels use synthetic display fixtures, and coverage curves use synthetic stored traces; no geometry cache or environment is needed for this style gate.
- Remaining corrections await their red/green checks and implementation.
- Correction 1 green: all 23 actual-drawing cases pass after the motion renderer uses the shared hollow-marker helper. Source commit 5272ba4 passed the full suite. The one affected motion figure is republished as v3 (PNG/PDF/SVG/JSON), with a separate manifest and unchanged v2 evidence. Its entire metadata equals v2 except the permitted render_commit; all original numeric/source fields are exact.
- Correction 2: installed Plotly 7.1.0 validates solid/dot/dash/dashdot, so the conditional stop rule did not trigger. Seven selected-method cases were observed red because line.dash was unset. The adapter now translates the committed base-style token for every selected and context trace; synthetic x/y data are asserted unchanged.

## Open issues

- Existing unknown source license versions and the viewer's documented default font remain outside this correction's scope.
